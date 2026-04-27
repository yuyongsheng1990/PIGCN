# -*- coding: utf-8 -*-


import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv


from torch.nn import Linear, BatchNorm1d, Sequential, ModuleList, ReLU, Dropout, ELU

from models.Attn_Head import SimpleAttnLayer
class GraphFormerBlock(nn.Module):
    '''
    adopt this module when using mini-batch
    '''
    def __init__(self, in_dim, hid_dim, out_dim, heads) -> None:  # in_dim, 302; hid_dim, 128; out_dim, 64, heads, 4
        super(GraphFormerBlock, self).__init__()
        self.GraphFormer1 = TransformerConv(in_channels=in_dim, out_channels=hid_dim, heads=heads, dropout=0.2, beta=True)
        self.GraphFormer2 = TransformerConv(in_channels=hid_dim * heads, out_channels=out_dim, dropout=0.2, beta=True)
        self.layers = ModuleList([self.GraphFormer1, self.GraphFormer2])
        self.norm = BatchNorm1d(heads * hid_dim)

    def forward(self, x, adjs, device):
        for i, (edge_index, _, size) in enumerate(adjs):
            # x: Tensor, edge_index: Tensor
            x, edge_index = x.to(device), edge_index.to(device)  # x: (2703, 302); (2, 53005); -> x: (1418, 512); (2, 2329)
            x_target = x[:size[1]]  # (1418, 302); (100, 512) Target nodes are always placed first;
            x = self.layers[i]((x, x_target), edge_index)  # (1418, 512) out_dim * heads; layers[2] output (100, 64)

            if i == 0:
                x = self.norm(x)
                x = F.elu(x)
                x = F.dropout(x, training=self.training)
            del edge_index
        return x

# GraphFormer model
class GraphFormer(nn.Module):
    '''
    inputs_list=feas_list, nb_classes=nb_classes, nb_nodes=nb_nodes, attn_drop=0.5,
                              ffd_drop=0.0, biases_list=biases_list, hid_units=args.hid_units, n_heads=args.n_heads,
                             residual=args.residual)

    '''
    def __init__(self, feature_size, hid_dim, out_dim, num_relations):
        super(GraphFormer, self).__init__()
        self.feature_size = feature_size  # list:3, (4762, 300)
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_relations = num_relations  # list:3, (4762,4762)
        self.n_heads = [2, 1, 4]  # [8,1]
        self.num_layers = 3
        self.norm = BatchNorm1d(self.hid_dim)

        self.intra_aggs = nn.ModuleList([GraphFormerBlock(self.feature_size, self.hid_dim, self.out_dim, self.n_heads[0])
                                         for _ in range(self.num_relations)])


    def forward(self, features, batch_nodes, adjs, n_ids, device):
        embed_list = []
        features = features.to(device)
        # multi-head attention in a hierarchical manner
        for i in range(self.num_relations):
            '''
            (n_ids[i])  # (2596,); (137,); (198,)
            edge_index=tensor([[]]), e_id=None, size=(2596,1222); edge_index=tensor([[]]), e_id=None, size=(1222, 100)
            edge_index=tensor([[]]), e_id=None, size=(137,129); edge_index=tensor([[]]), e_id=None, size=(129, 100)
            edge_index=tensor([[]]), e_id=None, size=(198,152); edge_index=tensor([[]]), e_id=None, size=(152, 100)
            '''
            # ---------------2 GraphFormer_layers-------------------------------------
            final_embedding = self.intra_aggs[i](features[n_ids[i]], adjs[i], device)  # (100,256) for DHAN4-2

            embed_list.append(final_embedding)

        features = torch.stack(embed_list, dim=0)  # (3, 100, 64)
        features = torch.transpose(features, dim0=0, dim1=1)  # (100, 3, 64)
        final_embed = features.reshape(len(batch_nodes), -1)  # (100, 192)
        gc.collect()

        return final_embed