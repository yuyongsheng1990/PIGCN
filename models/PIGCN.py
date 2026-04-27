# -*- coding: utf-8 -*-

import gc
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.PIGCNConv import PIGCNConv
from models.Attn_Head import SimpleAttnLayer


class PIGCNLayer(nn.Module):
    def __init__(self, feat_dim, hid_dim, out_dim, bias=True, learnable_k=False):
        super(PIGCNLayer, self).__init__()
        self.conv1 = PIGCNConv(in_channels=feat_dim, out_channels=hid_dim, k2=0.5, learnable_k=True, calculate_loss=True, gamma=0.5, learnable_gamma=True, bias=True)
        self.conv2 = PIGCNConv(in_channels=hid_dim, out_channels=out_dim, k2=0.5, learnable_k=True, calculate_loss=False, gamma=0.5, learnable_gamma=True, bias=True)


        self.norm = nn.BatchNorm1d(hid_dim)
        self.norm2 = nn.BatchNorm1d(out_dim)

        self.act = nn.Tanh()  # SiLU, Softplus(), Tanh.
        self.dropout = nn.Dropout()

    def forward(self, features, edge_index, batch_nodes, device):
        x, edge_index = features.to(device), edge_index.to(device)  # (4796, 302)

        x, helm_loss1 = self.conv1(x, edge_index)
        x = self.norm(x)
        x = self.act(x)
        x_force = self.dropout(x)

        x = self.conv2(x_force, edge_index, force_mode='gaussian')  # (4793, 128);
        # x, helm_loss2 = self.conv2(x_force, edge_index, force_mode='gaussian')  # (4793, 128);
        x = self.norm2(x)
        x_helm = self.act(x)

        return F.log_softmax(x_helm[batch_nodes], dim=1), helm_loss1 # (helm_loss1 + helm_loss2) /2
        # return F.log_softmax(x_helm[batch_nodes], dim=1), 1-F.cosine_similarity(x_force, x_helm, dim=-1), helm_loss1 # (helm_loss1 + helm_loss2) /2

# PIGCN
class PIGCN(nn.Module):
    '''
    inputs_list=feas_list, nb_classes=nb_classes, nb_nodes=nb_nodes, attn_drop=0.5,
                              ffd_drop=0.0, biases_list=biases_list, hid_units=args.hid_units, n_heads=args.n_heads,
                             residual=args.residual)
    '''
    def __init__(self, feature_size, hid_dim, out_dim, num_relations):
        super(PIGCN, self).__init__()
        self.feature_size = feature_size  # list:3, (4762, 300)
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_relations = num_relations  # list:3, (4762,4762)
        self.dropout = nn.Dropout()

        self.gcn_layers = nn.ModuleList([PIGCNLayer(self.feature_size, self.hid_dim, self.out_dim)
                                         for _ in range(self.num_relations)])  # DHAN-4: original version
        # -------------------meta-path aggregation----------------------------------
        self.simpleAttnLayer = SimpleAttnLayer(self.out_dim, self.hid_dim,  return_alphas=True)  # 64, 128

    def forward(self, features, multi_r_data, batch_nodes, device):
        embed_list = []
        features = features.to(device)
        helm_losses = []
        cos_losses = []

        for i, (edge_index) in enumerate(multi_r_data):
            # ---------------2 GCN layers from PIGCN-------------------------------------
            gcn_embedding, helm_loss = self.gcn_layers[i](features, edge_index, batch_nodes, device)  # (100,256), HelmholtzGCN
            # gcn_embedding, cos_loss, helm_loss = self.gcn_layers[i](features, edge_index, batch_nodes, device)  # (100,256), HelmholtzGCN
            embed_list.append(torch.unsqueeze(gcn_embedding, dim=1))
            # cos_losses.append(cos_loss)
            helm_losses.append(helm_loss)


        multi_embed = torch.cat(embed_list, dim=1)   # tensor, (100, 3, 64)
        # simple attention merge meta-based homo-graph embedding
        # final_embed, att_val = self.simpleAttnLayer(multi_embed, device)  # PIGCA, (100, 64)
        final_embed = multi_embed.view(batch_nodes.shape[0], -1)  # PIGCN
        del multi_embed
        gc.collect()

        # force physical force output and helm_loss alignment.
        # cos_loss_mean = torch.stack(cos_losses).mean()

        # helmholtzGCN: helm_losses is a list of Tensors -> stack and mean
        helm_loss_mean = torch.stack(helm_losses).mean()

        return final_embed, helm_loss_mean
        # return final_embed, cos_loss_mean, helm_loss_mean