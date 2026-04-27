# -*- coding: utf-8 -*-

import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


# HelmholtzGCN: PDE-Net: f(x) = alpha * self.tanh(x) + (1 - alpha) * self.softplus(x)
class TanhSoftplus(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(self, x):
        return self.alpha * self.tanh(x) + (1 - self.alpha) * self.softplus(x)

class RelGCNLayer(nn.Module):
    def __init__(self, feat_dim, hid_dim, out_dim):
        super(RelGCNLayer, self).__init__()
        self.conv1 = GCNConv(in_channels=feat_dim, out_channels=hid_dim)
        self.conv2 = GCNConv(in_channels=hid_dim, out_channels=out_dim)

        self.norm = nn.BatchNorm1d(hid_dim)
        self.norm2 = nn.BatchNorm1d(out_dim)

        self.act = nn.ReLU()
        self.dropout = nn.Dropout()

    def mish(self, x):
        return x * torch.tanh(F.softplus(x))

    def forward(self, features, edge_index, batch_nodes, device):
        x, edge_index = features.to(device), edge_index.to(device)  # (4796, 302)
        x = self.conv1(x, edge_index)  # GCN_normal(4793, 256)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)  # (4793, 128)
        x = self.norm2(x)
        x = self.act(x)

        return F.log_softmax(x[batch_nodes], dim=1)

# DoubleGCN model with RL
class DoubleGCN(nn.Module):
    '''
    inputs_list=feas_list, nb_classes=nb_classes, nb_nodes=nb_nodes, attn_drop=0.5,
                              ffd_drop=0.0, biases_list=biases_list, hid_units=args.hid_units, n_heads=args.n_heads,
                             residual=args.residual)
    '''
    def __init__(self, feature_size, hid_dim, out_dim, num_relations):
        super(DoubleGCN, self).__init__()
        self.feature_size = feature_size  # list:3, (4762, 300)
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_relations = num_relations  # list:3, (4762,4762)
        self.dropout = nn.Dropout()

        self.gcn_layers = nn.ModuleList([RelGCNLayer(self.feature_size, self.hid_dim, self.out_dim)
                                         for _ in range(self.num_relations)])

    def forward(self, features, multi_r_data, batch_nodes, device):
        embed_list = []

        features = features.to(device)

        for i, (edge_index) in enumerate(multi_r_data):

            # ---------------2 GCN layers-------------------------------------
            gcn_embedding = self.gcn_layers[i](features, edge_index, batch_nodes, device)  # (100,256)

            embed_list.append(torch.unsqueeze(gcn_embedding, dim=1))

        multi_embed = torch.cat(embed_list, dim=1)   # tensor, (100, 3, 64)
        final_embed = multi_embed.view(batch_nodes.shape[0], -1)
        del multi_embed
        gc.collect()

        return final_embed