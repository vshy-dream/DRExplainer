import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool as gmp, global_mean_pool
from GCNConv import GCNConv
from SGConv import SGConv
from torch.nn import Parameter
from feature_utils import *
import pandas as pd
import torch
EPS = 1e-15

class NodeRepresentation(nn.Module):
    def __init__(self, gcn_layer, dim_gexp, dim_methy, output, units_list=[256, 256, 256], use_relu=True, use_bn=True,
                 use_GMP=True, use_mutation=True, use_gexpr=True, use_copy_number=True):
        super(NodeRepresentation, self).__init__()
        torch.manual_seed(0)
        # Here modify the dimensions of the generated feature
        self.feature_dim = 64
        # -------drug layers
        self.use_relu = use_relu
        self.use_bn = use_bn
        self.units_list = units_list
        self.use_GMP = use_GMP

        self.use_mutation = use_mutation
        self.use_gexpr = use_gexpr
        self.use_copy_number = use_copy_number

        self.conv1 = SGConv(gcn_layer, units_list[0])
        self.batch_conv1 = nn.BatchNorm1d(units_list[0])
        self.graph_conv = []
        self.graph_bn = []
        for i in range(len(units_list) - 1):
            self.graph_conv.append(SGConv(units_list[i], units_list[i + 1]))
            self.graph_bn.append(nn.BatchNorm1d((units_list[i + 1])))
        self.conv_end = SGConv(units_list[-1], output)
        self.batch_end = nn.BatchNorm1d(output)
        # --------cell line layers (three omics)
        # -------gexp_layer
        self.fc_gexp1 = nn.Linear(dim_gexp, 256)
        self.batch_gexp1 = nn.BatchNorm1d(256)
        self.fc_gexp2 = nn.Linear(256, output)
        # -------methy_layer
        self.fc_methy1 = nn.Linear(dim_methy, 256)
        self.batch_methy1 = nn.BatchNorm1d(256)
        self.fc_methy2 = nn.Linear(256, output)
        # -------mut_layer
        self.cov1 = nn.Conv2d(1, 50, (1, 50), stride=(1, 5))
        self.cov2 = nn.Conv2d(50, 960, (1, 5), stride=(1, 2))
        self.fla_mut = nn.Flatten()
        self.fc_mut = nn.Linear(960, output)#The 30 here should be the same as the 30 of the cov2 output
        # ------Concatenate_three omics
        self.fcat = nn.Linear(self.feature_dim*3, output)
        self.batchc = nn.BatchNorm1d(self.feature_dim)
        self.reset_para()

    def reset_para(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        return

    def forward(self, epoch, epochs, drug_feature, drug_adj, ibatch, mutation_data, gexpr_data, copy_number_data):
        # -----drug representation
        x_drug = self.conv1(drug_feature, drug_adj)
        x_drug = F.relu(x_drug)
        x_drug = self.batch_conv1(x_drug)
        for i in range(len(self.units_list) - 1):
            x_drug = self.graph_conv[i](x_drug, drug_adj)
            x_drug = F.relu(x_drug)
            x_drug = self.graph_bn[i](x_drug)
        x_drug = self.conv_end(x_drug, drug_adj)
        x_drug = F.relu(x_drug)
        x_drug = self.batch_end(x_drug)
        if self.use_GMP:
            x_drug = gmp(x_drug, ibatch)
        else:
            x_drug = global_mean_pool(x_drug, ibatch)
        # save x_drug
        if epoch+1 == epochs or (epoch+1)%1000==0:
            print(f"This is epoch{epoch+1} and  x_drug is saved")
            x_drug_df = x_drug.detach().numpy()
            x_drug_df = pd.DataFrame(x_drug_df)
            x_drug_df.to_csv(f"../../data/node_representation/x_drug_{self.feature_dim}_epoch{epoch+1}.csv", index=False)

        # -----cell line representation
        # -----mutation representation
        if self.use_mutation:
            x_mutation = torch.tanh(self.cov1(mutation_data))
            x_mutation = F.max_pool2d(x_mutation, (1, 5))

            x_mutation = F.relu(self.cov2(x_mutation))
            x_mutation = F.max_pool2d(x_mutation, (1, 10))
            x_mutation = self.fla_mut(x_mutation)
            x_mutation = F.relu(self.fc_mut(x_mutation))

        # ----gexpr representation
        if self.use_gexpr:
            x_gexpr = torch.sigmoid(self.fc_gexp1(gexpr_data))
            x_gexpr = self.batch_gexp1(x_gexpr)
            x_gexpr = F.relu(self.fc_gexp2(x_gexpr))

        # ----methylation representation
        if self.use_copy_number:
            x_copy_number = torch.tanh(self.fc_methy1(copy_number_data))
            x_copy_number = self.batch_methy1(x_copy_number)
            x_copy_number = F.relu(self.fc_methy2(x_copy_number))

        # ------Concatenate representations of three omics
        if self.use_gexpr == False:
            x_cell = torch.cat((x_mutation, x_copy_number), 1)
        elif self.use_mutation == False:
            x_cell = torch.cat((x_gexpr, x_copy_number), 1)
        elif self.use_copy_number == False:
            x_cell = torch.cat((x_mutation, x_gexpr), 1)
        else:
            x_cell = torch.cat((x_mutation, x_gexpr, x_copy_number), 1)
        x_cell = F.leaky_relu(self.fcat(x_cell))
        # save x_cell
        if epoch+1 == epochs or (epoch+1)%1000==0:
            print(f"This is epoch{epoch+1} and x_cell is saved")
            x_cell_df = x_cell.detach().numpy()
            x_cell_df = pd.DataFrame(x_cell_df)
            x_cell_df.to_csv(f"../../data/node_representation/x_cell_{self.feature_dim}_epoch{epoch+1}.csv", index=False)
        # combine representations of cell line and drug
        x_all = torch.cat((x_cell, x_drug), 0)
        x_all = self.batchc(x_all)
        # save x_all
        if epoch + 1 == epochs or (epoch+1)%1000==0:
            print(f"This is epoch{epoch+1} and x_all is saved")
            x_all_df = pd.DataFrame(x_all.detach().numpy())
            x_all_df.to_csv(f"../../data/node_representation/x_all_{self.feature_dim}_epoch{epoch+1}.csv", index=False)
        return x_all

class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(Encoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=True)
        self.prelu1 = nn.PReLU(hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index.long())
        x = self.prelu1(x)
        return x

class Summary(nn.Module):
    def __init__(self, ino, inn):
        super(Summary, self).__init__()
        self.fc1 = nn.Linear(ino + inn, 1)

    def forward(self, xo, xn):
        m = self.fc1(torch.cat((xo, xn), 1))
        m = torch.tanh(torch.squeeze(m))
        m = torch.exp(m) / (torch.exp(m)).sum()
        x = torch.matmul(m, xn)
        return x

class GenerateFeature(nn.Module):
    def __init__(self, hidden_channels, encoder, summary, feat, index):
        super(GenerateFeature, self).__init__()
        # Here modify the dimensions of the generated feature
        self.feature_dim = 64
        self.hidden_channels = hidden_channels
        self.encoder = encoder
        self.summary = summary
        self.feat = feat
        self.index = index
        self.weight = Parameter(torch.Tensor(hidden_channels, hidden_channels))
        self.act = nn.Sigmoid()
        self.fc = nn.Linear(self.feature_dim, 64) #The 64 here is outputchannal
        self.fd = nn.Linear(self.feature_dim, 64) #The 64 here is outputchannal
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.summary)
        glorot(self.weight)
        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, epoch,epochs,drug_feature, drug_adj, ibatch, mutation_data, gexpr_data, methylation_data, edge):
        # ---CDR_graph_edge and corrupted CDR_graph_edge
        pos_edge = torch.from_numpy(edge[edge[:, 2] == 1, 0:2].T)
        neg_edge = torch.from_numpy(edge[edge[:, 2] == -1, 0:2].T)
        # ---cell+drug node attributes
        feature = self.feat(epoch,epochs,drug_feature, drug_adj, ibatch, mutation_data, gexpr_data, methylation_data)
        # ---cell+drug embedding from the CDR graph and the corrupted CDR graph
        pos_z = self.encoder(feature, pos_edge)
        neg_z = self.encoder(feature, neg_edge)
        # ---graph-level embedding (summary)
        summary_pos = self.summary(feature, pos_z)
        summary_neg = self.summary(feature, neg_z)
        # ---embedding at layer l
        cellpos = pos_z[:self.index, ];
        drugpos = pos_z[self.index:, ]
        # ---embedding at layer 0
        cellfea = self.fc(feature[:self.index, ]);
        drugfea = self.fd(feature[self.index:, ])
        cellfea = torch.sigmoid(cellfea);
        drugfea = torch.sigmoid(drugfea)
        # ---concatenate embeddings at different layers (0 and l)
        cellpos = torch.cat((cellpos, cellfea), 1)
        drugpos = torch.cat((drugpos, drugfea), 1)
        # ---inner product
        pos_adj = torch.matmul(cellpos, drugpos.t())
        pos_adj = self.act(pos_adj)
        return pos_z, neg_z, summary_pos, summary_neg, pos_adj.view(-1)

    def discriminate(self, z, summary, sigmoid=True):
        value = torch.matmul(z, torch.matmul(self.weight, summary))
        return torch.sigmoid(value) if sigmoid else value

    def loss(self, pos_z, neg_z, summary):
        pos_loss = -torch.log(
            self.discriminate(pos_z, summary, sigmoid=True) + EPS).mean()
        neg_loss = -torch.log(
            1 - self.discriminate(neg_z, summary, sigmoid=True) + EPS).mean()
        return pos_loss + neg_loss

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.hidden_channels)
