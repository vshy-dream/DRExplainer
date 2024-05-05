#!/usr/bin/env python3
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Lambda
import pandas as pd
import os
import numpy as np
import random as rn
from keras.callbacks import Callback
from prediction.code import utils
# import utils

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
class DGCN_Layer(tf.keras.layers.Layer):
    def __init__(self, num_entities, num_relations, output_dim, seed, **kwargs):
        super(DGCN_Layer, self).__init__(**kwargs)
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.output_dim = output_dim
        self.seed = seed

    def build(self, input_shape):
        input_dim = int(input_shape[-2][-1])

        self.relation_kernel = self.add_weight(
            shape=(self.num_relations, input_dim, self.output_dim),
            name="relation_kernels",
            trainable=True,
            initializer=tf.keras.initializers.RandomNormal(
                mean=0.0,
                stddev=1,
                seed=self.seed
            )
        )

        self.self_kernel = self.add_weight(
            shape=(input_dim, self.output_dim),
            name="self_kernel",
            trainable=True,
            initializer=tf.keras.initializers.RandomNormal(
                mean=0.0,
                stddev=1,
                seed=self.seed
            )
        )

    def call(self, inputs):
        embeddings, head_idx, head_e, tail_idx, tail_e, *adj_mats = inputs
        head_output = tf.matmul(head_e, self.self_kernel)
        tail_output = tf.matmul(tail_e, self.self_kernel)

        for i in range(self.num_relations):
            adj_i = tf.sparse.reshape(adj_mats[0][i], shape=(self.num_entities, self.num_entities))

            sum_embeddings = tf.sparse.sparse_dense_matmul(adj_i, embeddings)

            head_idx = tf.cast(head_idx, tf.int32)
            tail_idx = tf.cast(tail_idx, tf.int32)

            head_update = tf.nn.embedding_lookup(sum_embeddings, head_idx)
            tail_update = tf.nn.embedding_lookup(sum_embeddings, tail_idx)

            head_output += tf.matmul(head_update, self.relation_kernel[i])
            tail_output += tf.matmul(tail_update, self.relation_kernel[i])
        return tf.sigmoid(head_output), tf.sigmoid(tail_output)


class DistMult(tf.keras.layers.Layer):
    def __init__(self, num_relations, seed, **kwargs):
        super(DistMult, self).__init__(**kwargs)
        self.num_relations = num_relations
        self.seed = seed

    def build(self, input_shape):
        embedding_dim = input_shape[0][-1]

        self.kernel = self.add_weight(
            shape=(self.num_relations, embedding_dim),
            trainable=True,
            initializer=tf.keras.initializers.RandomNormal(
                mean=0.0,
                stddev=1,
                seed=self.seed
            ),
            name='rel_embedding'
        )

    def call(self, inputs):
        head_e, rel_idx, tail_e = inputs

        rel_idx = tf.cast(rel_idx, tf.int32)

        rel_e = tf.nn.embedding_lookup(self.kernel, rel_idx)

        score = tf.sigmoid(tf.reduce_sum(head_e * rel_e * tail_e, axis=-1))
        return tf.expand_dims(score, axis=0)

class DGCN_Model(tf.keras.Model):
    def __init__(self, num_entities,seed,mode,fold, *args, **kwargs):
        super(DGCN_Model, self).__init__(*args, **kwargs)
        self.num_entities = num_entities
        self.seed = seed
        self.mode = mode
        self.fold = fold

    def train_step(self, data):
        all_indices, pos_head, rel, pos_tail, *adj_mats = data[0]
        y_pos_true = data[1]
        X_train_neg = np.load(f'../data/split_data/mode{self.mode}_fold{self.fold}_X_train_neg.npy')
        num_samples = X_train_neg.shape[1]

        random_indices = np.random.permutation(num_samples)
        X_train_neg = X_train_neg[:, random_indices, :]
        neg_head = X_train_neg[:,:,0]
        neg_tail = X_train_neg[:,:,2]
        neg_rel = X_train_neg[:,:,1]
        neg_head = tf.convert_to_tensor(neg_head, dtype=tf.int64)
        neg_tail = tf.convert_to_tensor(neg_tail, dtype=tf.int64)
        neg_rel = tf.convert_to_tensor(neg_rel, dtype=tf.int64)

        with tf.GradientTape() as tape:
            y_pos_pred = self([
                all_indices,
                pos_head,
                rel,
                pos_tail,
                adj_mats
            ],training=True)
            y_neg_pred = self([
                all_indices,
                neg_head,
                neg_rel,
                neg_tail,
                adj_mats
            ],training=True)

            y_pred = tf.concat([y_pos_pred, y_neg_pred], axis=1)
            y_true = tf.concat([y_pos_true, tf.zeros_like(y_neg_pred)], axis=1)

            loss = self.compiled_loss(y_true, y_pred)

            loss *= (1 / self.num_entities)

            grads = tape.gradient(loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            self.compiled_metrics.update_state(y_pos_true, y_pos_pred)
            return {m.name: m.result() for m in self.metrics}

class SaveWeightsCallback(Callback):
    def __init__(self, save_epochs, save_path_template, mode, fold, learning_rate, batch_size, EMBEDDING_DIM):
        super(SaveWeightsCallback, self).__init__()
        self.save_epochs = save_epochs
        self.save_path_template = save_path_template
        self.mode = mode
        self.fold = fold
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.EMBEDDING_DIM = EMBEDDING_DIM

    def on_epoch_end(self, epoch, logs=None):
        if epoch + 1 in self.save_epochs:

            filename = self.save_path_template.format(mode=self.mode, fold=self.fold, epoch=epoch + 1,
                                                      learning_rate=self.learning_rate, batch_size=self.batch_size,
                                                      EMBEDDING_DIM=self.EMBEDDING_DIM)
            self.model.save_weights(filename)
            print(f"\nSaved weights for epoch {epoch + 1} to {filename}")

def get_DGCN_Model(num_entities,num_relations, embedding_dim, output_dim, seed, all_feature_matrix,mode,fold):
    head_input = tf.keras.Input(shape=(None,), name='head_input', dtype=tf.int64)
    rel_input = tf.keras.Input(shape=(None,), name='rel_input', dtype=tf.int64)
    tail_input = tf.keras.Input(shape=(None,), name='tail_input', dtype=tf.int64)
    all_entities = tf.keras.Input(shape=(None,), name='all_entities', dtype=tf.int64)

    adj_inputs = [tf.keras.Input(
        shape=(num_entities, num_entities),
        dtype=tf.float32,
        name='adj_inputs_' + str(i),
        sparse=True,
    ) for i in range(num_relations)]

    entity_embeddings = Embedding(
        input_dim=num_entities,
        output_dim=embedding_dim,
        name='entity_embeddings',
        weights=[tf.constant(all_feature_matrix, dtype=tf.float32)],
        trainable=True
        )
    head_e = entity_embeddings(head_input)
    tail_e = entity_embeddings(tail_input)
    all_e = entity_embeddings(all_entities)

    head_e = Lambda(lambda x: x[0, :, :])(head_e)
    tail_e = Lambda(lambda x: x[0, :, :])(tail_e)
    all_e = Lambda(lambda x: x[0, :, :])(all_e)
    head_index = Lambda(lambda x: x[0, :])(head_input)
    rel_index = Lambda(lambda x: x[0, :])(rel_input)
    tail_index = Lambda(lambda x: x[0, :])(tail_input)

    new_head, new_tail = DGCN_Layer(
        num_relations=num_relations,
        num_entities=num_entities,
        output_dim=output_dim,
        seed=seed)([
        all_e,
        head_index,
        head_e,
        tail_index,
        tail_e,
        adj_inputs])
    new_head, new_tail = DGCN_Layer(
        num_relations=num_relations,
        num_entities=num_entities,
        output_dim=output_dim,
        seed=seed)([
        all_e,
        head_index,
        new_head,
        tail_index,
        new_tail,
        adj_inputs])

    output = DistMult(num_relations=num_relations, seed=seed, name='DistMult')([new_head, rel_index, new_tail])

    model = DGCN_Model(
        inputs=[all_entities, head_input, rel_input, tail_input] + adj_inputs,
        outputs=[output],
        num_entities=num_entities,
        seed=seed,
        mode=mode,
        fold=fold
    )
    return model


if __name__ == '__main__':

    SEED = 89
    os.environ['PYTHONHASHSEED'] = str(SEED)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    rn.seed(SEED)

    batch_size = [256]
    learning_rate = [0.001]
    embedding_dim = [64]
    save_epochs = [1000,3000,5000]

    for bs in batch_size:
        for lr in learning_rate:
            for ed in embedding_dim:
                BATCH_SIZE = bs
                LEARNING_RATE = lr
                NUM_EPOCHS = 5000
                EMBEDDING_DIM = ed

                OUTPUT_DIM = EMBEDDING_DIM
                NUM_ENTITIES = 634
                NUM_RELATIONS = 4

                resopnse_pairs = pd.read_csv('../data/resopnse_triples.csv', header=0)
                cell_similar_triples = pd.read_csv('../data/cell_similar_triples.csv', header=0)
                drug_similar_triples = pd.read_csv('../data/drug_similar_triples.csv', header=0)

                X_train_neg = pd.read_csv('../data/negative_dc.csv', header=0)
                X_train_neg = X_train_neg.sample(frac=1,random_state=SEED).reset_index(drop=True)

                # Set up a 5% fold cross-validation for resopnse_pairs
                resopnse_pairs.columns = ['obj', 'rel', 'sbj']
                cell_similar_triples.columns = ['obj', 'rel', 'sbj']
                drug_similar_triples.columns = ['obj', 'rel', 'sbj']

                num_splits = 5

                # mode0 is a normal experiment, mode1 is a no_cell, mode2 is a no_drug, and mode3 is a no_cell_drug
                for mode in range(0,1):

                    train_test_splits = utils.split_pos_triple_into_folds(resopnse_pairs, cell_similar_triples, drug_similar_triples, num_folds=num_splits, seed=SEED, mode=mode)
                    neg_train_test_splits = utils.split_neg_triple_into_folds(X_train_neg, num_folds=num_splits, seed=SEED, mode=mode)

                    for fold in range(0,1):

                        X_train_response, X_test_response = train_test_splits[fold]
                        neg_X_train, neg_X_test = neg_train_test_splits[fold]

                        neg_X_test.to_csv(f'../data/split_data/mode{mode}_fold{fold}_neg_X_test.csv',index_label=None)

                        X_train_triple = X_train_response
                        X_test_triple = X_test_response

                        X_test_triple  = X_test_triple.drop(X_test_triple[(X_test_triple['rel'] == 2) | (X_test_triple['rel'] == 3)].index).copy()
                        syn_X_train_triple = pd.DataFrame(utils.generate_reverse_triplets(X_train_triple.to_numpy()))
                        syn_neg_X_train = pd.DataFrame(utils.generate_reverse_triplets(neg_X_train.to_numpy()))
                        syn_X_test_triple = pd.DataFrame(utils.generate_reverse_triplets(X_test_triple.to_numpy()))
                        syn_neg_X_test = pd.DataFrame(utils.generate_reverse_triplets(neg_X_test.to_numpy()))

                        syn_X_train_triple.columns = X_train_triple.columns;syn_X_test_triple.columns = X_test_triple.columns
                        syn_neg_X_train.columns = ['obj', 'rel', 'sbj'];syn_neg_X_test.columns=['obj', 'rel', 'sbj']

                        neg_X_train = pd.concat([neg_X_train,syn_neg_X_train],axis=0)
                        neg_X_test = pd.concat([neg_X_test,syn_neg_X_test],axis=0)

                        X_train = pd.concat([X_train_triple, syn_X_train_triple], axis=0).astype(np.int64)
                        X_test = pd.concat([X_test_triple, syn_X_test_triple],axis=0).astype(np.int64)

                        X_train.to_csv(f"../data/split_data/mode{mode}_fold{fold}_X_train.csv", index=False)
                        X_test.to_csv(f"../data/split_data/mode{mode}_fold{fold}_X_test.csv", index=False)

                        all_feature_matrix = pd.read_csv(f"../data/node_representation/x_all_{EMBEDDING_DIM}.csv", header=0)

                        ADJ_MATS = utils.get_adj_mats(X_train.values, NUM_ENTITIES, NUM_RELATIONS)

                        X_train = np.expand_dims(X_train, axis=0)

                        X_train_neg = np.expand_dims(neg_X_train, axis=0)
                        np.save(f'../data/split_data/mode{mode}_fold{fold}_X_train_neg.npy', X_train_neg)

                        ALL_INDICES = np.arange(NUM_ENTITIES).reshape(1, -1)

                        model = get_DGCN_Model(
                            num_entities=NUM_ENTITIES,
                            num_relations=NUM_RELATIONS,
                            embedding_dim=EMBEDDING_DIM,
                            output_dim=OUTPUT_DIM,
                            seed=SEED,
                            all_feature_matrix=all_feature_matrix,
                            mode=mode,
                            fold=fold
                        )
                        # Zeros the model weights before each loop iteration of the fold
                        model.reset_states()

                        model.compile(
                            loss=tf.keras.losses.BinaryCrossentropy(),
                            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                        )

                        save_path_template = os.path.join('../data', 'weights', 'mode{mode}_fold{fold}_epoch{epoch}_learnRate{learning_rate}_batchsize{batch_size}_embdim{EMBEDDING_DIM}.h5')
                        save_weights_callback = SaveWeightsCallback(save_epochs=save_epochs, save_path_template=save_path_template, mode=mode, fold=fold, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, EMBEDDING_DIM=EMBEDDING_DIM)

                        history = model.fit(
                            x=[
                                ALL_INDICES,
                                X_train[:, :, 0],
                                X_train[:, :, 1],
                                X_train[:, :, 2],
                                ADJ_MATS
                            ],
                            y=np.ones(X_train.shape[1]).reshape(1, -1),
                            epochs=NUM_EPOCHS,
                            batch_size=BATCH_SIZE,
                            verbose=1,
                            callbacks=[save_weights_callback]
                        )

                        print('len(X_train_response)',len(X_train_response))
                        print('len(X_train),len(X_test)', len(X_train[0]), len(X_test))
                        print(f'len(neg_X_train),len(neg_X_test): ',len(neg_X_train),len(neg_X_test))
                        print(f'Done mode{mode}_fold{fold}')