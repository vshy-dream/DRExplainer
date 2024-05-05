import tensorflow as tf
import numpy as np
import sys
sys.path.append("../prediction")
from prediction.code import utils
import random as rn
from prediction.code import DGCN
import pandas as pd
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def get_neighbors(data_subset,node_idx):
    head_neighbors = tf.boolean_mask(data_subset,data_subset[:,0]==node_idx)
    tail_neighbors = tf.boolean_mask(data_subset,data_subset[:,2]==node_idx)
    neighbors = tf.concat([head_neighbors,tail_neighbors],axis=0)
    return neighbors
def get_computation_graph(head,rel,tail,data,num_relations):
    '''Get k hop neighbors (may include duplicates)'''
    neighbors_head = get_neighbors(data,head)
    neighbors_tail = get_neighbors(data,tail)
    all_neighbors = tf.concat([neighbors_head,neighbors_tail],axis=0)
    return all_neighbors

def replica_step(head, rel, tail, num_entities, num_relations):
    comp_graph = get_computation_graph(head, rel, tail, ADJACENCY_DATA, num_relations)
    adj_mats = utils.get_adj_mats(comp_graph, num_entities, num_relations)
    total_loss = 0.0
    for epoch in range(NUM_EPOCHS):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(masks)
            masked_adjs = [adj_mats[i] * tf.sigmoid(masks[i]) for i in range(num_relations)]
            before_pred = model([
                ALL_INDICES,
                tf.reshape(head, (1, -1)),
                tf.reshape(rel, (1, -1)),
                tf.reshape(tail, (1, -1)),
                adj_mats
            ])
            pred = model([
                ALL_INDICES,
                tf.reshape(head, (1, -1)),
                tf.reshape(rel, (1, -1)),
                tf.reshape(tail, (1, -1)),
                masked_adjs
            ])
            loss = - before_pred * tf.math.log(pred + 0.00001)
            tf.print(f"current loss {loss}")
            total_loss += loss
        grads = tape.gradient(loss, masks)
        optimizer.apply_gradients(zip(grads, masks))

    current_pred = []
    current_scores = []
    for i in range(num_relations):
        mask_i = adj_mats[i] * tf.nn.sigmoid(masks[i])
        mask_idx = mask_i.values > THRESHOLD
        non_masked_indices = tf.gather(mask_i.indices[mask_idx], [1, 2], axis=1)
        if tf.reduce_sum(non_masked_indices) != 0:
            rel_indices = tf.cast(tf.ones((non_masked_indices.shape[0], 1)) * i, tf.int64)
            triple = tf.concat([non_masked_indices, rel_indices], axis=1)
            triple = tf.gather(triple, [0, 2, 1], axis=1)
            score_array = mask_i.values[mask_idx]
            current_pred.append(triple)
            current_scores.append(score_array)
    current_scores = tf.concat([array for array in current_scores], axis=0)
    top_k_scores = tf.argsort(current_scores, direction='DESCENDING')[0:10]
    pred_exp = tf.reshape(tf.concat([array for array in current_pred], axis=0), (-1, 3))
    pred_exp = tf.gather(pred_exp, top_k_scores, axis=0)
    for mask in masks:
        mask.assign(value=init_value)
    return total_loss, pred_exp
def distributed_replica_step(head,rel,tail,num_entities, num_relations):
    per_replica_losses, current_preds = strategy.run(replica_step,args=(head,rel,tail,num_entities, num_relations))
    reduce_loss = per_replica_losses / NUM_EPOCHS
    return reduce_loss, current_preds

if __name__ == '__main__':

    SEED = 123
    os.environ['PYTHONHASHSEED'] = str(SEED)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    rn.seed(SEED)

    NUM_EPOCHS = 20
    EMBEDDING_DIM = 64
    LEARNING_RATE = 0.001
    MAX_PADDING = 2

    NUM_ENTITIES = 634
    NUM_RELATIONS = 4
    OUTPUT_DIM = EMBEDDING_DIM
    THRESHOLD = 0.01

    for fold in range(0,1):
        train2idx = pd.read_csv(fr'../prediction/data/split_data/mode0_fold{fold}_X_train.csv', header=0, dtype=int).values
        test2idx = pd.read_csv(fr'data/test_filtered_fold{fold}.csv', header=0, dtype=int).values

        ALL_INDICES = tf.reshape(tf.range(0,NUM_ENTITIES,1,dtype=tf.int64), (1,-1))

        strategy = tf.distribute.MirroredStrategy()
        print(f'Number of devices: {strategy.num_replicas_in_sync}')

        with strategy.scope():

            all_feature_matrix = pd.read_csv(r"../prediction/data/node_representation/x_all_64.csv", header=0)

            model = DGCN.get_DGCN_Model(
                num_entities=NUM_ENTITIES,
                num_relations=NUM_RELATIONS,
                embedding_dim=EMBEDDING_DIM,
                output_dim=OUTPUT_DIM,
                seed=SEED,
                all_feature_matrix=all_feature_matrix,
                mode=0,
                fold=fold
            )

            model.load_weights(os.path.join(f'../prediction/data/weights/mode0_fold{fold}_epoch5000_learnRate0.001_batchsize256_embdim64_weight.h5'))
            optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
            init_value = tf.random.normal(
                    (1,NUM_ENTITIES,NUM_ENTITIES),
                    mean=0,
                    stddev=1,
                    dtype=tf.float32,
                    seed=SEED
                )
            masks = [tf.Variable(
                initial_value=init_value,
                name='mask_'+str(i),
                trainable=True) for i in range(NUM_RELATIONS)
            ]

        ADJACENCY_DATA = tf.concat([train2idx,test2idx,],axis=0)

        del train2idx

        best_preds = []
        tf_data = tf.data.Dataset.from_tensor_slices((test2idx[:,0],test2idx[:,1],test2idx[:,2])).batch(1)
        dist_dataset = strategy.experimental_distribute_dataset(tf_data)

        for head,rel,tail in dist_dataset:
            loss, current_preds = distributed_replica_step(head,rel,
                tail,NUM_ENTITIES, NUM_RELATIONS)
            best_preds.append(current_preds)
        best_preds = [array.numpy() for array in best_preds]
        out_preds = []

        for i in range(len(best_preds)):
            preds_i = utils.idx2array(best_preds[i])
            out_preds.append(preds_i)

        out_preds = np.array(out_preds,dtype=object)
        print(f'Num epochs: {NUM_EPOCHS}')
        print(f'Embedding dim: {EMBEDDING_DIM}')
        print(f'learning_rate: {LEARNING_RATE}')
        print(f'threshold {THRESHOLD}')

        np.savez(f'data/DRExplainer_preds_fold{fold}.npz',preds=out_preds)
        print('Done.')

