import pandas as pd
def dataload(copy_number_file,expression_file,mutation_file,pub_smiles_file,GDSC_file):

    copy_number_feature = pd.read_csv(copy_number_file,sep=',',header=0,index_col=[0])
    gexpr_feature = pd.read_csv(expression_file,sep=',',header=0,index_col=[0])
    mutation_feature = pd.read_csv(mutation_file,sep=',',header=0,index_col=[0])
    pub_smiles = pd.read_csv(pub_smiles_file,header=0)
    GDSC = pd.read_csv(GDSC_file,header=0)

    GDSC['PubChem_ID'] = GDSC['PubChem_ID'].astype(int)
    GDSC['Y'] = GDSC['Y'].astype(int)

    data_idx = []
    for tup in zip(GDSC['CELL_LINE_NAME'],GDSC['PubChem_ID'],GDSC['Y']):
        # Take only the edges that contain the drug in Pubsmile and the edges that contain the cells in the cell line
        if tup[1] in pub_smiles['pubchems'].values and tup[0] in copy_number_feature.index:
            data_idx.append((tup[0],str(tup[1]),tup[2]))

    #----eliminate ambiguity responses
    data_sort=sorted(data_idx, key=(lambda x: [x[0], x[1], x[2]]), reverse=True)
    data_tmp=[];data_new=[];data_duplicate_i=[];data_duplicate_k=[]
    data_idx1 = [[i[0],i[1]] for i in data_sort]
    for i,k in zip(data_idx1,data_sort):
        if i not in data_tmp:
            data_tmp.append(i)
            data_new.append(k)
        else:
            data_duplicate_i.append(i)
            data_duplicate_k.append(k)

    nb_celllines = len(set([item[0] for item in data_new]))
    nb_drugs = len(set([item[1] for item in data_new]))
    print('All %d pairs across %d cell lines and %d drugs.'%(len(data_new),nb_celllines,nb_drugs))

    triples = pd.DataFrame(data_new).to_numpy()
    # Swap columns 2 and 3
    triples[:, [1, 2]] = triples[:, [2, 1]]
    triples = pd.DataFrame(triples)
    triples.to_csv("../../data/response_triples_entity.csv",index=False)
    return pub_smiles, mutation_feature, gexpr_feature, copy_number_feature, data_new, nb_celllines, nb_drugs


