import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

for fold in range(0,1):
    test_df = pl.read_csv(f'../prediction/data/split_data/mode0_fold{fold}_X_test.csv').to_pandas()
    gt_all = pl.read_csv(f'data/gt_all.csv').to_pandas().dropna(how='all')

    print(f'This is fold{fold},the shape of the raw GT is{gt_all.shape}')
    # Match the corresponding gt for the test set triplet
    gt_fold = test_df.merge(gt_all, on=['obj', 'rel', 'sbj'], how='left')
    print('The shape of the data after matching the folded test set:',gt_fold.shape)

    # Remove gt with less than 10 bars
    gt_fold['sub_triplet_count'] = gt_fold.iloc[:, 3:].apply(lambda x: (~pd.isnull(x)).sum(), axis=1)/3
    max_sub_triplet_count = gt_fold['sub_triplet_count'].max()
    print("The maximum number of GTs per edge:", int(max_sub_triplet_count))

    # Filter out rows with more than 10 triplets of interpretation
    gt_filtered = gt_fold[gt_fold['sub_triplet_count'] >= 10]
    test_filtered = gt_filtered[['obj', 'rel', 'sbj']]
    print("The shape of the GT is filtered out for rows with more than 10 GTs:",gt_filtered.iloc[:,:-1].shape)

    # Plot a histogram to count the frequency of occurrence of GT numbers
    num_bins = 20
    bins = np.linspace(gt_filtered['sub_triplet_count'].min(), gt_filtered['sub_triplet_count'].max(), num_bins + 1)
    labels = [f'{int(bins[i])}-{int(bins[i+1])-1}' for i in range(len(bins)-1)]
    gt_filtered['sub_triplet_group'] = pd.cut(gt_filtered['sub_triplet_count'], bins=bins, labels=labels, right=False)
    frequency = gt_filtered['sub_triplet_group'].value_counts().sort_index()

    plt.figure(figsize=(10, 12))
    plt.bar(frequency.index, frequency.values)
    plt.xlabel('Number of gt',fontsize=14)
    plt.xticks(rotation=45,fontsize=12)
    plt.yticks(fontsize=12)
    plt.subplots_adjust(top=0.9)
    plt.ylabel('Frequency',fontsize=14)
    plt.title('Distribution of explain_triplets Number',fontsize=12)
    plt.savefig(f'data/fold{fold}_plot.png',dpi=300,bbox_inches='tight')
    # plt.show()
    print('Converting a rowed gt to an ndarray list triplet......')

    # The GT arranged by row is converted into an NDarray list triplet
    gt_filtered_np = gt_filtered.iloc[:, 3:-1].to_numpy()
    num_rows = len(gt_filtered_np)
    gt_fold = [[gt_filtered_np[i, j:j+3] for j in range(0, gt_filtered_np.shape[1], 3)] for i in range(num_rows)]
    print('The NAN value is being removed……')
    gt_fold = [[tri.astype(float).tolist() for tri in gt if not np.any(np.isnan(tri.astype(float)))] for gt in gt_fold]

    np.savez(f'data/gt_filtered_fold{fold}.npz',result=np.array(gt_fold, dtype=object))
    test_filtered.to_csv(f'data/test_filtered_fold{fold}.csv',index=False)