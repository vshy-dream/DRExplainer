import time
from model import *
from data_process import process
import argparse
from feature_utils import *
from data_load import dataload
import torch

parser = argparse.ArgumentParser(description='Drug_response_pre')
parser.add_argument('--alph', dest='alph', type=float, default=0.30, help='')
parser.add_argument('--beta', dest='beta', type=float, default=0.30, help='')
parser.add_argument('--epoch', dest='epoch', type=int, default=10000, help='')
parser.add_argument('--hidden_channels', dest='hidden_channels', type=int, default=256, help='')
# If you want to change the dimension of the generated feature, you need to change the two feature_dim in the model.py
parser.add_argument('--output_channels', dest='output_channels', type=int, default=64, help='')
args = parser.parse_args()
start_time = time.time()

#------data files
copy_number_file = '../../data/original_data/copy_number.csv'
expression_file = '../../data/original_data/expression.csv'
mutation_file = '../../data/original_data/mutation.csv'
pub_smiles_file = '../../data/original_data/isosmiles.csv'
GDSC_file = '../../data/original_data/GDSC.csv'

#-------bio-feature extraction
pub_smiles, mutation_feature, gexpr_feature, copy_number_feature, data_new, nb_celllines, nb_drugs = dataload(copy_number_file,expression_file,mutation_file,pub_smiles_file,GDSC_file)

# -------split train and test sets
drug_set,cellline_set,train_edge,label_pos,train_mask,test_mask,atom_shape = process(pub_smiles, mutation_feature, gexpr_feature, copy_number_feature, data_new, nb_celllines, nb_drugs)


model = GenerateFeature(hidden_channels=args.hidden_channels, encoder=Encoder(args.output_channels, args.hidden_channels), summary=Summary(args.output_channels, args.hidden_channels),
                        feat=NodeRepresentation(atom_shape,gexpr_feature.shape[-1],copy_number_feature.shape[-1],args.output_channels), index=nb_celllines)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
myloss = nn.BCELoss()

def train(epoch,epochs):
    model.train()
    loss_temp=0
    for batch, (drug,cell) in enumerate(zip(drug_set,cellline_set)):
        optimizer.zero_grad()
        pos_z, neg_z, summary_pos, summary_neg, pos_adj=model(epoch,epochs,drug.x, drug.edge_index, drug.batch, cell[0], cell[1], cell[2], train_edge)
        dgi_pos = model.loss(pos_z, neg_z, summary_pos)
        dgi_neg = model.loss(neg_z, pos_z, summary_neg)
        pos_loss = myloss(pos_adj[train_mask],label_pos[train_mask])
        loss=(1-args.alph-args.beta)*pos_loss + args.alph*dgi_pos + args.beta*dgi_neg
        loss.backward()
        optimizer.step()
        loss_temp += loss.item()
    print('train loss: ', str(round(loss_temp, 4)))

def test():
    model.eval()
    with torch.no_grad():
        for batch, (drug, cell) in enumerate(zip(drug_set, cellline_set)):
            _, _, _, _, pre_adj=model(epoch,epochs,drug.x, drug.edge_index, drug.batch,cell[0], cell[1], cell[2], train_edge)
            loss_temp = myloss(pre_adj[test_mask],label_pos[test_mask])
        yp=pre_adj[test_mask].detach().numpy()
        ytest=label_pos[test_mask].detach().numpy()
        AUC, AUPR, F1, ACC =metrics_graph(ytest,yp)
        print('test loss: ', str(round(loss_temp.item(), 4)))
        print('test auc: ' + str(round(AUC, 4)) + '  test aupr: ' + str(round(AUPR, 4)) +
              '  test f1: ' + str(round(F1, 4)) + '  test acc: ' + str(round(ACC, 4)))
    return AUC, AUPR, F1, ACC

#------main
final_AUC = 0;final_AUPR = 0;final_F1 = 0;final_ACC = 0;epochs = args.epoch
for epoch in range(args.epoch):

    print('\nepoch: ' + str(epoch))
    train(epoch,args.epoch)
    AUC, AUPR, F1, ACC = test()
    if (AUC > final_AUC):
        final_AUC = AUC;final_AUPR = AUPR;final_F1 = F1;final_ACC = ACC

elapsed = time.time() - start_time
print('---------------------------------------')
print('Elapsed time: ', round(elapsed, 4))
print('Final_AUC: ' + str(round(final_AUC, 4)) + '  Final_AUPR: ' + str(round(final_AUPR, 4)) +
      '  Final_F1: ' + str(round(final_F1, 4)) + '  Final_ACC: ' + str(round(final_ACC, 4)))
print('---------------------------------------')
