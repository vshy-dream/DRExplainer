import numpy as np
import pandas as pd
import os
import utils
import random as rn
import tensorflow as tf
import DGCN
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import precision_recall_curve
import csv

SEED = 123
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
tf.random.set_seed(SEED)
np.random.seed(SEED)
rn.seed(SEED)

mode = 0
folds = [0]
epoch = 5000
lr = 0.001
bs = 256
emb_dim = 64

NUM_ENTITIES = 634
NUM_RELATIONS = 4

for fold in folds:
    OUTPUT_DIM = emb_dim
    X_train = pd.read_csv(f"../data/split_data/mode{mode}_fold{fold}_X_train.csv")
    X_test_pos= pd.read_csv(f"../data/split_data/mode{mode}_fold{fold}_X_test.csv",header=0,index_col=None)
    # Generate negative sample edges
    heads = X_test_pos['obj'];relations=X_test_pos['rel'];tails=X_test_pos['sbj']

    ALL_INDICES = np.arange(NUM_ENTITIES).reshape(1,-1)

    x_all = pd.read_csv(f"../data/node_representation/x_all_{emb_dim}.csv", header=0).values
    x_all_test = utils.extract_feature_in_x_all(X_test_pos, x_all)

    model = DGCN.get_DGCN_Model(
        num_entities=NUM_ENTITIES,
        num_relations=NUM_RELATIONS,
        embedding_dim=emb_dim,
        output_dim=OUTPUT_DIM,
        seed=SEED,
        all_feature_matrix=x_all,
        mode=mode,
        fold=fold
    )

    model.load_weights(os.path.join('../data', 'weights', f'mode{mode}_fold{fold}_epoch{epoch}_learnRate{lr}_batchsize{bs}_embdim{emb_dim}.h5'))

    ADJACENCY_DATA = tf.concat([X_train,X_test_pos], axis=0)

    ADJ_MATS = utils.get_adj_mats(ADJACENCY_DATA, NUM_ENTITIES, NUM_RELATIONS)

    X_test_neg= pd.read_csv(f'../data/split_data/mode{mode}_fold{fold}_neg_X_test.csv',header=0,index_col=0)
    X_test_neg.columns = X_test_pos.columns
    X_test = pd.concat([X_test_pos,X_test_neg], axis=0)

    X_test = np.expand_dims(X_test,axis=0)
    X_test_pos = np.expand_dims(X_test_pos,axis=0)

    RULES = [0,1]
    rel2idx = {0:0,1:1}
    threshold = 0.5

    # Indicators of sensitivity and resistance are calculated separately
    for rule in RULES:
        rule_indices = X_test[0,:,1] == rel2idx[rule]
        X_test_rule = X_test[:, rule_indices, :]
        preds = model.predict(
            x=[
                ALL_INDICES,
                X_test_rule[:, :, 0],
                X_test_rule[:, :, 1],
                X_test_rule[:, :, 2],
                ADJ_MATS
            ]
        )

        y_pred = np.zeros_like(preds)
        y_pred[preds > threshold] = 1
        y_prob = preds[0]
        y_true = utils.get_y_true(X_test_pos[0], X_test_rule[0])

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred[0]).ravel()
        acc = (tn + tp) / (tn + fp + fn + tp)
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        specificity = tn / (tn +fp)
        f1 = 2 * precision * recall / (precision + recall)

        roc_auc = roc_auc_score(y_true, y_prob)
        prec, reca, _ = precision_recall_curve(np.array(y_true), np.array(y_prob))
        aupr = auc(reca, prec)
        print(f' -----------relation{rule}\naccuracy:{acc:.4f}')
        print(f'tp:{tp} | tn:{tn} | fp:{fp} | fn:{fn} | recall:{recall:.4f} | precision:{precision:.4f} | specificity:{specificity:.4f} | f1:{f1:.4f}')
        print(f'roc_auc:{roc_auc:.4f} | aupr:{aupr:.4f}\n')

    # Indicators of a mixture of sensitivity and resistance
    rule_indices0 = X_test[0, :, 1] == rel2idx[0]
    rule_indices1 = X_test[0, :, 1] == rel2idx[1]
    rule_indices = np.logical_or(rule_indices0, rule_indices1)
    X_test_rule = X_test[:, rule_indices, :]

    preds = model.predict(
        x=[
            ALL_INDICES,
            X_test_rule[:, :, 0],
            X_test_rule[:, :, 1],
            X_test_rule[:, :, 2],
            ADJ_MATS
        ]
    )

    y_pred = np.zeros_like(preds)
    y_pred[preds > threshold] = 1
    y_prob = preds[0]

    y_true = utils.get_y_true(X_test_pos[0], X_test_rule[0])

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred[0]).ravel()
    acc = (tn + tp) / (tn + fp + fn + tp)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    specificity = tn / (tn + fp)

    f1 = 2 * precision * recall / (precision + recall)

    # np.save(f'../../plot/DGCN/y_true_mode{mode}_fold{fold}.npy', y_true)
    # np.save(f'../../plot/DGCN/y_prob_mode{mode}_fold{fold}.npy', y_prob)

    roc_auc = roc_auc_score(y_true, y_prob)
    prec, reca, _ = precision_recall_curve(np.array(y_true), np.array(y_prob))
    aupr = auc(reca, prec)
    print(f'relation01\naccuracy:{acc:.4f}')
    print(f'tp:{tp} | tn:{tn} | fp:{fp} | fn:{fn} | recall:{recall:.4f} | precision:{precision:.4f} | specificity:{specificity:.4f}  | f1:{f1:.4f}')
    print(f'roc_auc:{roc_auc:.4f} | aupr:{aupr:.4f}\n')

    # # Write the results to a file
    # with open('../data/performance.csv', 'a', encoding='utf-8', newline='') as fa:
    #     writer = csv.writer(fa)
    #     if fa.tell() == 0:
    #         writer.writerow(
    #             ['Rel','Mode', 'Fold', 'Epoch', 'LR', 'BS', 'EMBEDDING_DIM', 'TP', 'TN', 'FP', 'FN', 'Accuracy', 'Recall',
    #              'Precision', 'Specificity', 'F1 Score', 'ROC AUC', 'AUPR'])
    #
    #     writer.writerow([10,mode,fold,epoch,lr,bs,emb_dim,
    #                      tp, tn, fp, fn,
    #                      acc,
    #                      recall, precision, specificity,
    #                      f1,
    #                      roc_auc, aupr])