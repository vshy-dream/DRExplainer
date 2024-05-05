import pandas as pd
import numpy as np
from tqdm import tqdm
import csv
import sys
sys.path.append("../prediction")
from prediction.code import utils

fold = 0
explain_id = 1

gt_fold = np.load(f'data/gt_filtered_fold{fold}.npz', allow_pickle=True)['result']
if explain_id==1:
    # 1 is used for DRExplainer and 0 for explaiNE
    explaine_preds = np.load(f'data/DRExplainer_preds_fold{fold}.npz', allow_pickle=True)['preds']
elif explain_id==0:
    explaine_preds = np.load(f'data/explaiNE_preds_fold{fold}.npz', allow_pickle=True)['preds']


# Metrics are calculated on an item-by-item basis
metrics = []
for index, preds in tqdm(enumerate(explaine_preds),total=len(explaine_preds)):
    preds = utils.remove_syn_triples(preds)
    preds_set = set(map(tuple, preds))

    preds_flip = np.array([row[::-1] for row in preds])
    preds_flip_set = set(map(tuple, preds_flip))

    preds_set_top5 = set(map(tuple, preds[:5]))
    preds_flip_set_top5 = set(map(tuple, preds_flip[:5]))

    gt_fold_set = set(map(tuple, gt_fold[index]))

    precision, recall, f1_score = utils.calculate_metrics(preds_set,preds_flip_set,preds_set_top5,preds_flip_set_top5, gt_fold_set)
    metrics.append((precision, recall, f1_score))

metrics_df = pd.DataFrame(metrics, columns=['Precision@5', 'Recall@5', 'F1@5'])

avr_prec = metrics_df['Precision@5'].mean()
avr_rec = metrics_df['Recall@5'].mean()
avr_f1 = metrics_df['F1@5'].mean()

print(metrics_df)
print(f'Total average Precision@5: {avr_prec:.4f}')
print(f'Total average Recall@5: {avr_rec:.4f}')
print(f'Total average F1@5: {avr_f1:.4f}')

# with open('data/explain_performance.csv', 'a', encoding='utf-8', newline='') as fa:
#     writer = csv.writer(fa)
#     if fa.tell() == 0:
#         writer.writerow(['Fold', 'explain_id', 'avr_prec','avr_rec', 'avr_f1'])
#     writer.writerow([fold, explain_id, f'{avr_prec:.4f}', f'{avr_rec:.4f}', f'{avr_f1:.4f}'])