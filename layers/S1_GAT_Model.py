# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch.nn import Linear, BatchNorm1d, ModuleList, ReLU, Dropout

class GAT(nn.Module):
    '''
    adopt this module when using mini-batch
    '''

    def __init__(self, in_dim, hid_dim, out_dim, heads) -> None:  # in_dim, 302; hid_dim, 128; out_dim, 64, heads, 4
        super(GAT, self).__init__()
        self.GAT1 = GATConv(in_channels=in_dim, out_channels=hid_dim, heads=heads,
                            add_self_loops=False)
        self.GAT2 = GATConv(in_channels=hid_dim * heads, out_channels=out_dim, add_self_loops=False)
        self.layers = ModuleList([self.GAT1, self.GAT2])
        self.norm = BatchNorm1d(heads * hid_dim)

    def forward(self, x, adjs, device):
        for i, (edge_index, _, size) in enumerate(adjs):
            # x: Tensor, edge_index: Tensor
            x, edge_index = x.to(device), edge_index.to(device)  # x: (2703, 302); (2, 53005); -> x: (1418, 512); (2, 2329)
            x_target = x[:size[1]]
            if i == 0:
                x = self.layers[i]((x, x_target), edge_index)
                x = self.norm(x)
                x = F.elu(x)
                x = F.dropout(x, training=self.training)
            else:
                x = self.layers[i]((x, x_target), edge_index)
            del edge_index
        return x


# GAT model, intra means inside, which is exactly opposite of extra
class Intra_AGG(nn.Module):  # intra-aggregation
    def __init__(self, GAT_args):
        super(Intra_AGG, self).__init__()
        in_dim, hid_dim, out_dim, heads = GAT_args
        self.gnn = GAT(in_dim, hid_dim, out_dim, heads)  # in_dim, 302; hid_dim, 128; out_dim, 64, heads, 4

    def forward(self, x, adjs, device):
        x = self.gnn(x, adjs, device)
        return x  # (100, 64)


# mlp model, inter means middle
class Inter_AGG(nn.Module):  # inter-aggregation
    def __init__(self, mlp_args=None):
        super(Inter_AGG, self).__init__()
        if mlp_args is not None:
            hid_dim, out_dim = mlp_args
            self.mlp = nn.Sequential(
                Linear(hid_dim, hid_dim),
                BatchNorm1d(hid_dim),
                ReLU(inplace=True),
                Dropout(),
                Linear(hid_dim, out_dim),
            )

    def forward(self, features, thresholds, inter_opt):  # (3,100,64), [[0.2], [0.2], [0.2]], 'cat_w_avg'
        batch_size = features[0].size(0)  # 100
        features = torch.transpose(features, dim0=0, dim1=1)  # (100, 3, 64)
        if inter_opt == 'cat_wo_avg':
            features = features.reshape(batch_size, -1)
        elif inter_opt == 'cat_w_avg':
            # weighted average and concatenate
            features = torch.mul(features, thresholds).reshape(batch_size, -1)  # (100, 192)
        elif inter_opt == 'cat_w_avg_mlp':
            features = torch.mul(features, thresholds).reshape(batch_size, -1)
            features = self.mlp(features)
        elif inter_opt == 'cat_wo_avg_mlp':
            features = torch.mul(features, thresholds).reshape(batch_size, -1)
            features = self.mlp(features)
        elif inter_opt == 'add_wo_avg':
            features = features.sum(dim=1)
        elif inter_opt == 'add_w_avg':
            features = torch.mul(features, thresholds).sum(dim=1)
        return features
