# -*- coding: utf-8 -*-


import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.PFGCNConv import PFGCNConv

from models.Attn_Head import SimpleAttnLayer

class PFGCNLayer(nn.Module):
    def __init__(self, feat_dim, hid_dim, out_dim):
        super(PFGCNLayer, self).__init__()
        self.conv1 = PFGCNConv(in_channels=feat_dim, out_channels=hid_dim, gamma=0.5, learnable_gamma=True)
        self.conv2 = PFGCNConv(in_channels=hid_dim, out_channels=out_dim,  gamma=0.5, learnable_gamma=True)

        self.norm = nn.BatchNorm1d(hid_dim)
        self.norm2 = nn.BatchNorm1d(out_dim)
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()

    def forward(self, features, edge_index, batch_nodes, device, warm_up=False):
        x, edge_index = features.to(device), edge_index.to(device)  # (4796, 302)
        x = self.conv1(x, edge_index)  # (4793, 256)
        x = self.norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index, force_mode='gaussian')  # (4793, 128)
        x = self.norm2(x)
        x = self.relu(x)

        return F.log_softmax(x[batch_nodes], dim=1)

# GCN with Physical Force
class PFGCN(nn.Module):
    '''
    inputs_list=feas_list, nb_classes=nb_classes, nb_nodes=nb_nodes, attn_drop=0.5,
                              ffd_drop=0.0, biases_list=biases_list, hid_units=args.hid_units, n_heads=args.n_heads,
                             residual=args.residual)
    '''
    def __init__(self, feature_size, hid_dim, out_dim, num_relations):
        super(PFGCN, self).__init__()
        self.feature_size = feature_size  # list:3, (4762, 300)
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_relations = num_relations  # list:3, (4762,4762)
        self.hid_units = [4]
        self.n_heads = [2, 1, 4]  # [8,1]
        self.num_layers = 3
        self.elu = nn.ELU()
        self.softplus = nn.Softplus()
        self.dropout = nn.Dropout()

        self.gcn_layers = nn.ModuleList([PFGCNLayer(self.feature_size, self.hid_dim, self.out_dim)
                                         for _ in range(self.num_relations)])
        # -------------------meta-path aggregation----------------------------------
        self.simpleAttnLayer = SimpleAttnLayer(self.out_dim, self.hid_dim,  return_alphas=True)  # 64, 128


    def forward(self, features, multi_r_data, batch_nodes, device, warm_up=False):

        # original embeddings: [4762, 302]
        features = features.to(device)

        embed_list = []
        # multi-head attention in a hierarchical manner
        for i, (edge_index) in enumerate(multi_r_data):

            # ---------------2 GCN layers------------------------------------
            gcn_embedding = self.gcn_layers[i](features, edge_index, batch_nodes, device, warm_up=warm_up)  # with training force, (100,256)

            embed_list.append(torch.unsqueeze(gcn_embedding, dim=1))

        multi_embed = torch.cat(embed_list, dim=1)   # tensor, (100, 3, 64)
        # final_embed, att_val = self.simpleAttnLayer(multi_embed, sin_vals[batch_nodes], cos_vals[batch_nodes], device)  # (100, 64)
        # PFGCN: mean aggregation.
        final_embed = multi_embed.view(batch_nodes.shape[0], -1)
        del multi_embed
        gc.collect()

        return final_embed