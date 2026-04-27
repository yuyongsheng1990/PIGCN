# -*- coding: utf-8 -*-


import gc
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.HelmholtzGCNConv import HelmholtzGCNConv


class TanhSoftplus(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(self, x):
        return self.alpha * self.tanh(x) + (1 - self.alpha) * self.softplus(x)

class HelmholtzGCNLayer(nn.Module):
    def __init__(self, feat_dim, hid_dim, out_dim):
        super(HelmholtzGCNLayer, self).__init__()
        self.conv1 = HelmholtzGCNConv(in_channels=feat_dim, out_channels=hid_dim, k2=0.5, learnable_k=True, calculate_loss=True, bias=True)
        self.conv2 = HelmholtzGCNConv(in_channels=hid_dim, out_channels=out_dim, k2=0.5, learnable_k=True, calculate_loss=False, bias=True)

        self.norm = nn.BatchNorm1d(hid_dim)
        self.norm2 = nn.BatchNorm1d(out_dim)

        self.act = nn.Tanh()  # SiLU, Softplus(), Tanh.
        # self.act = TanhSoftplus(alpha=0.5)
        self.dropout = nn.Dropout()


    def forward(self, features, edge_index, batch_nodes, device):
        x, edge_index = features.to(device), edge_index.to(device)  # (4796, 302)

        # x = self.conv1(x, edge_index)  # GCN_normal(4793, 256)        # x, helm_loss2 = self.conv2(x, edge_index)  # (4793, 12
        x, helm_loss1 = self.conv1(x, edge_index)  # (4793, 128)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)  # (4793, 128)
        # x, helm_loss2 = self.conv2(x, edge_index)  # (4793, 128)
        x = self.norm2(x)
        x = self.act(x)

        # return F.log_softmax(x[batch_nodes], dim=1)
        return F.log_softmax(x[batch_nodes], dim=1), helm_loss1 # (helm_loss1 + helm_loss2) /2


# HelmholtzGCN model
class HelmholtzGCN(nn.Module):

    def __init__(self, feature_size, hid_dim, out_dim, num_relations):
        super(HelmholtzGCN, self).__init__()
        self.feature_size = feature_size  # list:3, (4762, 300)
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_relations = num_relations  # list:3, (4762,4762)
        self.dropout = nn.Dropout()

        self.gcn_layers = nn.ModuleList([HelmholtzGCNLayer(self.feature_size, self.hid_dim, self.out_dim)
                                         for _ in range(self.num_relations)])

    def forward(self, features, multi_r_data, batch_nodes, device):
        embed_list = []
        features = features.to(device)
        helm_losses = []

        for i, (edge_index) in enumerate(multi_r_data):
            # gcn_embedding = self.gcn_layers[i](features, edge_index, batch_nodes, device)  # (100,256), HelmholtzGCN
            gcn_embedding, helm_loss = self.gcn_layers[i](features, edge_index, batch_nodes, device)  # (100,256), HelmholtzGCN
            embed_list.append(torch.unsqueeze(gcn_embedding, dim=1))
            helm_losses.append(helm_loss)


        multi_embed = torch.cat(embed_list, dim=1)   # tensor, (100, 3, 64)
        final_embed = multi_embed.view(batch_nodes.shape[0], -1)
        del multi_embed
        gc.collect()

        helm_loss_mean = torch.stack(helm_losses).mean()
        # return final_embed
        return final_embed, helm_loss_mean