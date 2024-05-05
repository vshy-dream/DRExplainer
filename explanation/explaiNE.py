import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import os
import sys
sys.path.append('../prediction')
from prediction.code import utils
import random as rn
from prediction.code import DGCN

def remove_padding_np(data, unk_ent_id, unk_rel_id):
    """
    Removes rows from a numpy array that contain 'unk_ent' or 'unk_rel' padding elements.

    :param data: numpy array from which to remove padding.
    :param unk_ent_id: Identifier for unknown entities.
    :param unk_rel_id: Identifier for unknown relations.
    :return: numpy array with padding removed.
    """
    # Find rows that do not contain either 'unk_ent' or 'unk_rel'
    rows_to_keep = ~((data == unk_ent_id) | (data == unk_rel_id)).any(axis=1)
    return data[rows_to_keep]

def get_pred(adj_mats, num_relations, tape, pred, top_k):
    scores = []

    for i in range(num_relations):

        adj_mat_i = adj_mats[i]

        for idx, score in enumerate(tape.gradient(pred, adj_mat_i.values).numpy()):
            if tf.abs(score) >= 0:
                scores.append((idx, i, score))

    top_k_scores = sorted(scores, key=lambda x: x[2], reverse=True)[:top_k]

    pred_triples = []

    for idx, rel, score in top_k_scores:
        indices = adj_mats[rel].indices.numpy()[idx, 1:]

        head, tail = indices

        pred_triple = [head, rel, tail]

        pred_triples.append(pred_triple)

    return np.array(pred_triples)


if __name__ == '__main__':

    SEED = 123

    os.environ['PYTHONHASHSEED'] = str(SEED)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    rn.seed(SEED)

    DATASET = 'all'
    RULE = 'full_data'
    TOP_K = 10
    EMBEDDING_DIM = 64

    MAX_PADDING = 2
    NUM_ENTITIES = 634
    NUM_RELATIONS = 4
    OUTPUT_DIM = EMBEDDING_DIM

    unk_ent_id = 'UNK_ENT',
    unk_rel_id = 'UNK_REL',

    ALL_INDICES = tf.reshape(tf.range(0, NUM_ENTITIES, 1, dtype=tf.int64), (1, -1))


    for fold in range(0,1):
        train2idx = pd.read_csv(fr'../prediction/data/split_data/mode0_fold{fold}_X_train.csv', header=0, dtype=int).values
        test2idx = pd.read_csv(fr'data/test_filtered_fold{fold}.csv', header=0, dtype=int).values

        ADJACENCY_DATA = tf.concat([
            train2idx,
            test2idx,
        ], axis=0
        )
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

        ADJ_MATS = utils.get_adj_mats(ADJACENCY_DATA, NUM_ENTITIES, NUM_RELATIONS)

        tf_data = tf.data.Dataset.from_tensor_slices(
            (test2idx[:, 0], test2idx[:, 1], test2idx[:, 2])).batch(1)

        pred_exps = []

        for head, rel, tail in tqdm(tf_data,total=len(tf_data)):
            with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
                tape.watch(ADJ_MATS)
                pred = model([
                    ALL_INDICES,
                    tf.reshape(head, (1, -1)),
                    tf.reshape(rel, (1, -1)),
                    tf.reshape(tail, (1, -1)),
                    ADJ_MATS
                ])
            pred_exp = get_pred(ADJ_MATS, NUM_RELATIONS, tape, pred, TOP_K)
            pred_exps.append(pred_exp)

        preds = np.array(pred_exps)

        np.savez(f'data/explaiNE_preds_fold{fold}.npz',preds=preds)

        print('Done.')
