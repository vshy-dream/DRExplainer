import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

x_cell = pd.read_csv("../data/node_representation/x_cell_64.csv", header=0)
x_drug = pd.read_csv("../data/node_representation/x_drug_64.csv", header=0)
def creat_similar_mat(mat, threshold):
# Set the similar elements in the similarity matrix to 1 and the rest to 0
    result_mat = np.zeros_like(mat)
    result_mat[mat > threshold] = 1
    result_mat.astype(int)
    return result_mat

def simat2triple(mat,relation,start):
# The similarity matrix is transformed into a triplet
    triples = []
    num_nodes = mat.shape[0]
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i < j and mat[i,j]!= 0:
                i1 = i+start;j1=j+start
                triples.append((i1,relation,j1))
    return triples

# Calculate cosine similarity
cell_dist_mat = cosine_similarity(x_cell)
drug_dist_mat = cosine_similarity(x_drug)

# Set a threshold for cosine similarity and create a similarity matrix
cell_threshold = 0.85
drug_threshold = 0.85

cell_similar_mat = creat_similar_mat(cell_dist_mat, cell_threshold).astype(int)
drug_similar_mat = creat_similar_mat(drug_dist_mat, drug_threshold).astype(int)

# Convert the similarity matrix to a triplet
cell_triples = pd.DataFrame(simat2triple(cell_similar_mat,relation=3,start=0))
drug_triples = pd.DataFrame(simat2triple(drug_similar_mat,relation=2,start=len(x_cell)))

cell_triples.to_csv(f'../data/similar_cell.csv', index=False, header=False)
drug_triples.to_csv(f'../data/similar_drug.csv', index=False, header=False)

print(f'len(cell_triples):{len(cell_triples)},len(drug_triples):{len(drug_triples)}')
print(f'Proportion of cells：{round(len(cell_triples)/477/477,3)}, Proportion of drugs：{round(len(drug_triples)/157/157,3)}')
