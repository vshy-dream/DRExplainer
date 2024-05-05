import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import KFold

def distinct(a):
    _a = np.unique(a, axis=0)
    return _a

def get_adj_mats(data, num_entities, num_relations):
    adj_mats = []
    for i in range(num_relations):

        data_i = data[data[:, 1] == i]

        if not data_i.shape[0]:
            indices = tf.zeros((1, 2), dtype=tf.int64)
            values = tf.zeros((indices.shape[0]))

        else:
            indices = tf.gather(data_i, [0, 2], axis=1)
            indices = tf.py_function(distinct, [indices], indices.dtype)
            indices = tf.dtypes.cast(indices, tf.int64)
            values = tf.ones((indices.shape[0]))

        sparse_mat = tf.sparse.SparseTensor(
            indices=indices,
            values=values,
            dense_shape=(num_entities, num_entities)
        )
        sparse_mat = tf.sparse.reorder(sparse_mat)
        sparse_mat = tf.sparse.reshape(sparse_mat, shape=(1, num_entities, num_entities))
        adj_mats.append(sparse_mat)
    return adj_mats

def extract_feature_in_x_all(X,all_feature_matrix):
    nodes_to_extract = set()
    for triplet in X.values:
        nodes_to_extract.add(triplet[0])
        nodes_to_extract.add(triplet[2])
    x_all_new = []
    for node in nodes_to_extract:
        x_all_new.append(all_feature_matrix[node])

    x_all_new = pd.DataFrame(x_all_new)
    return  x_all_new

def generate_reverse_triplets(triplets):
    # Create a list of reverse triples
    reverse_triplets = [(tail, relation, head) for head, relation, tail in triplets if head != tail]
    return np.array(reverse_triplets)

def extract_feature_in_x_all(X,x_all):
    # Extract the eigenvectors of node 1 and node 2
    nodes_to_extract = set()
    for triplet in X.values:
        nodes_to_extract.add(triplet[0])
        nodes_to_extract.add(triplet[2])
    x_all_new = []
    for node in nodes_to_extract:
        x_all_new.append(x_all[node])
    return pd.DataFrame(x_all_new)

def get_y_true(X_test_pos, X_test_rule):
    # The real side of the X_test_rule is marked as 1, and the negative sample side generated is marked as 0
    X_test_pos = pd.DataFrame(X_test_pos).drop_duplicates()
    X_test_rule = pd.DataFrame(X_test_rule)
    X_merged = pd.merge(X_test_rule, X_test_pos, indicator=True, how='left')
    y_true = (X_merged['_merge'] == 'both').astype(int)
    return y_true.values

def split_pos_triple_into_folds(dc, cc, dd, num_folds, seed, mode):

    dc = dc.sample(frac=1, random_state=seed).reset_index(drop=True)
    cc = cc.sample(frac=1, random_state=seed).reset_index(drop=True)
    dd = dd.sample(frac=1, random_state=seed).reset_index(drop=True)

    dc_cc_dd = pd.concat([dc, dd, cc], axis=0).astype(int)
    cc_dd = pd.concat([cc, dd], axis=0)

    if mode==0:
        kf = KFold(n_splits=num_folds)
        train_test_splits = []
        for train_index, test_index in kf.split(dc):
            train_data, test_data = dc.iloc[train_index], dc.iloc[test_index]
            train_data = pd.concat([train_data,cc_dd],axis=0)
            train_test_splits.append((train_data, test_data))

    elif mode==1:
        train_test_splits = []
        num_entity = 477
        len_fold = num_entity//num_folds
        np.random.seed(seed)
        for i in range(num_folds):
            nodes = range(int(len_fold*i), int(len_fold*(i+1)))
            test_df = dc_cc_dd[(dc_cc_dd['obj'].isin(nodes)) | (dc_cc_dd['sbj'].isin(nodes))].astype(int)
            merged_df = pd.merge(dc_cc_dd, test_df, how='left', indicator=True)
            train_df = merged_df[merged_df['_merge'] == 'left_only'].drop('_merge', axis=1).astype(int)

            train_test_splits.append((train_df, test_df))

    elif mode==2:
        train_test_splits = []
        num_entity = 157
        len_fold = num_entity/num_folds
        np.random.seed(seed)

        for i in range(num_folds):
            nodes = range(int(len_fold*i+477), int(len_fold*(i+1)+477))
            test_df = dc_cc_dd[(dc_cc_dd['obj'].isin(nodes)) | (dc_cc_dd['sbj'].isin(nodes))].astype(int)
            merged_df = pd.merge(dc_cc_dd, test_df, how='left', indicator=True)
            train_df = merged_df[merged_df['_merge'] == 'left_only'].drop('_merge', axis=1).astype(int)
            train_test_splits.append((train_df, test_df))
    else:
        train_test_splits = []
        num_entity = 477
        len_fold = num_entity/num_folds
        np.random.seed(seed)
        for i in range(num_folds):
            nodes = range(int(len_fold*i), int(len_fold*(i+1)))

            pre_test_df = dc_cc_dd[(dc_cc_dd['obj'].isin(nodes)) | (dc_cc_dd['sbj'].isin(nodes))]

            merged_df = pd.merge(dc_cc_dd, pre_test_df, how='left', indicator=True)
            pre_train_df = merged_df[merged_df['_merge'] == 'left_only'].drop('_merge', axis=1).astype(int)

            all_nodes = np.unique(pre_test_df[['obj', 'sbj']].values)
            new_node = list(np.setdiff1d(all_nodes, nodes))

            new_node_test = new_node[:len(new_node)//2]
            new_node_train = new_node[len(new_node)//2:]

            test_df = pre_test_df[~pre_test_df['obj'].isin(new_node_test) & ~pre_test_df['sbj'].isin(new_node_test)].astype(int)

            train_df = pre_train_df[~pre_train_df['obj'].isin(new_node_train) & ~pre_train_df['sbj'].isin(new_node_train)].astype(int)

            train_test_splits.append((train_df, test_df))

    return train_test_splits
def split_neg_triple_into_folds(dc, num_folds, seed, mode):
    dc = dc.sample(frac=1, random_state=seed).reset_index(drop=True)

    if mode==0:
        kf = KFold(n_splits=num_folds)
        train_test_splits = []
        for train_index, test_index in kf.split(dc):
            train_data, test_data = dc.iloc[train_index], dc.iloc[test_index]
            train_test_splits.append((train_data, test_data))

    elif mode==1:
        train_test_splits = []
        num_entity = 477
        len_fold = num_entity//num_folds
        np.random.seed(seed)
        for i in range(num_folds):
            nodes = range(int(len_fold*i), int(len_fold*(i+1)))
            test_df = dc[(dc['obj'].isin(nodes)) | (dc['sbj'].isin(nodes))].astype(int)

            merged_df = pd.merge(dc, test_df, how='left', indicator=True)
            train_df = merged_df[merged_df['_merge'] == 'left_only'].drop('_merge', axis=1).astype(int)

            train_test_splits.append((train_df, test_df))

    elif mode==2:
        train_test_splits = []
        num_entity = 157
        len_fold = num_entity/num_folds
        np.random.seed(seed)

        for i in range(num_folds):
            nodes = range(int(len_fold*i+477), int(len_fold*(i+1)+477))

            test_df = dc[(dc['obj'].isin(nodes)) | (dc['sbj'].isin(nodes))].astype(int)
            merged_df = pd.merge(dc, test_df, how='left', indicator=True)
            train_df = merged_df[merged_df['_merge'] == 'left_only'].drop('_merge', axis=1).astype(int)

            train_test_splits.append((train_df, test_df))
    else:
        train_test_splits = []
        num_entity = 477
        len_fold = num_entity/num_folds
        np.random.seed(seed)
        for i in range(num_folds):
            nodes = range(int(len_fold*i), int(len_fold*(i+1)))

            pre_test_df = dc[(dc['obj'].isin(nodes)) | (dc['sbj'].isin(nodes))]

            merged_df = pd.merge(dc, pre_test_df, how='left', indicator=True)
            pre_train_df = merged_df[merged_df['_merge'] == 'left_only'].drop('_merge', axis=1).astype(int)
            all_nodes = np.unique(pre_test_df[['obj', 'sbj']].values)
            new_node = list(np.setdiff1d(all_nodes, nodes))
            new_node_test = new_node[:len(new_node)//2]
            new_node_train = new_node[len(new_node)//2:]
            test_df = pre_test_df[~pre_test_df['obj'].isin(new_node_test) & ~pre_test_df['sbj'].isin(new_node_test)].astype(int)
            train_df = pre_train_df[~pre_train_df['obj'].isin(new_node_train) & ~pre_train_df['sbj'].isin(new_node_train)].astype(int)

            train_test_splits.append((train_df, test_df))

    return train_test_splits
def idx2array(dataset):
    if dataset.ndim == 2:
        data = []
        for head_idx, rel_idx, tail_idx in dataset:
            head = head_idx
            tail = tail_idx
            rel = rel_idx
            data.append((head, rel, tail))

        data = np.array(data)

    return data
def remove_syn_triples(preds):
    visited_indices = []
    to_remove_indices = []

    for i, pred1 in enumerate(preds):
        if i in visited_indices:
            continue
        for j, pred2 in enumerate(preds[i + 1:], start=i + 1):
            if np.array_equal(pred1[::-1], pred2):
                to_remove_indices.append(j)
                break
        visited_indices.append(i)

    filtered_preds = np.delete(preds, to_remove_indices, axis=0)

    return filtered_preds

def calculate_metrics(preds_set,preds_flip_set,preds_set_top5,preds_flip_set_top5, gt_fold_set):

    tp = len(set(preds_set_top5).intersection(gt_fold_set))+len(set(preds_flip_set_top5).intersection(gt_fold_set))
    fp = 5 - tp
    fn = 10 - len(set(preds_set).intersection(gt_fold_set))-len(set(preds_flip_set).intersection(gt_fold_set))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1_score