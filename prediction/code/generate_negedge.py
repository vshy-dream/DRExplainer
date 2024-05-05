import pandas as pd
import numpy as np
from tqdm import tqdm

response_triples = pd.read_csv("../data/response_triples_entity.csv", header=0).values.tolist()
cell_similar_triples = pd.read_csv("../data/similar_cell.csv", header=None)
drug_similar_triples = pd.read_csv("../data/similar_drug.csv", header=None)

cell_entities = list(set([item[0] for item in response_triples]));cell_entities.sort()
drug_entities = list(set([item[2] for item in response_triples]));drug_entities.sort()
cell_entities_map = list(zip(cell_entities, list(range(len(cell_entities)))))
drug_entities_map = list(zip(drug_entities, list(range(len(cell_entities), len(cell_entities) + len(drug_entities)))))
cell_entities_num = np.squeeze([[j[1] for j in cell_entities_map if i[0] == j[0]] for i in response_triples])
drug_entities_num = np.squeeze([[j[1] for j in drug_entities_map if i[2] == j[0]] for i in response_triples])
relation_num = np.squeeze([i[1] for i in response_triples])
resopnse_pairs = np.vstack((cell_entities_num, relation_num, drug_entities_num)).T
column_data = {0: resopnse_pairs[:, 0], 1: resopnse_pairs[:, 1], 2: resopnse_pairs[:, 2]}
resopnse_pairs = pd.DataFrame(column_data)
print(len(resopnse_pairs), len(cell_similar_triples), len(drug_similar_triples))

positive_df = pd.concat([resopnse_pairs, cell_similar_triples, drug_similar_triples], axis=0)


positive_df.columns = ['obj','rel','sbj']

# Generate all possible combinations of cells and drugs
cells = list(range(477))
drugs = list(range(477, 634))

negative_data = pd.DataFrame({'obj': [], 'rel': [], 'sbj': []})

negative_dc = negative_data; negative_cc = negative_data; negative_dd = negative_data

# Generate negative samples in space
for cell in tqdm(cells):
    for drug in drugs:
        if ((positive_df['obj'] == cell) & (positive_df['sbj'] == drug)).any() or \
           ((positive_df['obj'] == drug) & (positive_df['sbj'] == cell)).any():
            continue
        else:
            random_relation = np.random.choice([0, 1],p=[0.5,0.5])
            negative_dc = pd.concat([negative_dc, pd.DataFrame({'obj': [cell], 'rel': [random_relation], 'sbj': [drug]})])
negative_dc = negative_dc.sample(n=2*len(resopnse_pairs), random_state=45)

for cell1 in tqdm(cells):
    for cell2 in cells:
        # Check if the same triplet is present in the positive sample, If present, skip current medications
        if ((positive_df['obj'] == cell1) & (positive_df['sbj'] == cell2)).any() or \
           ((positive_df['obj'] == cell2) & (positive_df['sbj'] == cell1)).any():
            continue
        elif cell1 != cell2:
            random_relation = 3
            negative_cc = pd.concat([negative_cc, pd.DataFrame({'obj': [cell1], 'rel': [random_relation], 'sbj': [cell2]})])
negative_cc = negative_cc.sample(n=len(negative_dc), random_state=45)

for drug1 in tqdm(drugs):
    for drug2 in drugs:
        # Check if the same triplet is present in the positive sample, If present, skip current medications
        if ((positive_df['obj'] == drug1) & (positive_df['sbj'] == drug2)).any() or \
           ((positive_df['obj'] == drug2) & (positive_df['sbj'] == drug1)).any():
            continue
        elif drug1 != drug2:
            random_relation = 2
            negative_dd = pd.concat([negative_dd, pd.DataFrame({'obj': [drug1], 'rel': [random_relation], 'sbj': [drug2]})])
# negative_dd = negative_dd.sample(n=len(negative_dc), random_state=45)

negative_triples = pd.concat([negative_dc,negative_cc,negative_dd],axis=0)

resopnse_pairs.to_csv('../data/resopnse_triples.csv',index=None)
cell_similar_triples.to_csv('../data/cell_similar_triples.csv',index=None)
drug_similar_triples.to_csv('../data/drug_similar_triples.csv',index=None)

negative_dc.to_csv('../data/negative_dc.csv',index=None)
negative_dd.to_csv('../data/negative_dd.csv', index=None)
negative_cc.to_csv('../data/negative_cc.csv',index=None)
negative_triples.to_csv('../data/negative_triples.csv',index=None)
