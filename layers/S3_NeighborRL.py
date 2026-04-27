# -*- coding: utf-8 -*-


from typing import Any, Dict
import numpy as np
import torch
from torch.functional import Tensor
import math
import os

def cal_similarity_node_edge(multi_r_data, features, save_path=None):

    relation_config: Dict[str, Dict[int, Any]] = {}
    for relation_id, r_data in enumerate(multi_r_data):
        node_config: Dict[int, Any] = {}
        r_data: Tensor  # entity,(2, 487962); (2, 8050); (2, 51498)
        unique_nodes = r_data[1].unique()  # neighbor idx: entity, (4762, );
        num_nodes = unique_nodes.size(0)  # 4762
        for node in range(num_nodes):  # neighbor index
            # get neighbors' index
            neighbors_idx = torch.where(r_data[1] == node)[0]  # how many same neighbor, return neighbor index
            # get neghbors
            neighbors = r_data[0, neighbors_idx]
            num_neighbors = neighbors.size(0)  # node number
            neighbors_features = features[neighbors, :]  # different node embedding
            target_features = features[node, :]  # neighbor embedding
            # calculate enclidean distance with broadcast
            dist: Tensor = torch.norm(neighbors_features - target_features, p=2, dim=1)
            # smaller is better and we use 'top p' in our paper
            # (threshold * num_neighbors) see RL_neighbor_filter for details
            sorted_neighbors, sorted_index = dist.sort(descending=False)
            node_config[node] = {'neighbors_idx': neighbors_idx,
                                'sorted_neighbors': sorted_neighbors,
                                'sorted_index': sorted_index,
                                'num_neighbors': num_neighbors}
        relation_config['relation_%d' % relation_id] = node_config  # relation neighbor config
    if save_path is not None:
        print(save_path)
        save_path = os.path.join(save_path, 'relation_config.npy')
        np.save(save_path, relation_config)

def RL_neighbor_filter(multi_r_data, RL_thtesholds, load_path, model_name, device='cpu'):  # (2, 487962), (2, 8050), (2,51499)
    load_path = load_path + '/relation_config.npy'
    relation_config = np.load(load_path, allow_pickle=True)  # dict: 3. dict: 4762, neighbor similarity
    relation_config = relation_config.tolist()  # {0: neighbor_idx: tensor([0]), num_neighbors:1, sorted_index: tensor([0]); 1:{....}}
    relations = list(relation_config.keys())  # ['relation_0', 'relation_1', 'relation_2'], entity, userid, word
    multi_remain_data = []

    for i in range(len(relations)):  # 3, entity, userid, word
        edge_index: Tensor = multi_r_data[i]  # (2, 487962); (2, 8050); (2,51499), node -> neighbors
        unique_nodes = edge_index[1].unique()  # target neighbor node 4762
        num_nodes = unique_nodes.size(0)  # neighbor number, 4762
        num_nodes = torch.tensor(num_nodes).to(device)
        remain_node_index = torch.tensor([]).to(device)
        for node in range(num_nodes):
            # extract config，sorted neighbor nodes，sorted neighbor idx
            neighbors_idx = relation_config[relations[i]][node]['neighbors_idx'].to(device)
            filtered_neighbors_idx = torch.tensor([]).to(device)
            num_neighbors = relation_config[relations[i]][node]['num_neighbors']
            num_neighbors = torch.tensor(num_neighbors).to(device)
            sorted_neighbors = relation_config[relations[i]][node]['sorted_neighbors'].to(device)  #sorted similarity
            sorted_index = relation_config[relations[i]][node]['sorted_index'].to(device)  # sorted neighbor index

            if num_neighbors < 5:  # one agent
                remain_node_index = torch.cat((remain_node_index, neighbors_idx))
                continue  # add limitations

            if model_name == 'ReDHAN':  # 精确匹配
                if num_neighbors <= 10:  # second agent
                    threshold = float(RL_thtesholds[i]) + 0.3
                else:  # third agent
                    threshold = float(RL_thtesholds[i]) + 0.2
                num_kept_neighbors = min(math.ceil(num_neighbors * threshold) + 1, 8)
                filtered_neighbors_idx = neighbors_idx[
                    sorted_index[:num_kept_neighbors]]  # 注意这里的sort_index指的是最佳neighbor在neighbors_idx中的索引，草了！
            elif model_name in ['FinEvent', 'TGAT']:
            # # FinEvent model: AC
                threshold = float(RL_thtesholds[i])
                num_kept_neighbors = min(math.ceil(num_neighbors * threshold) + 1, 8)
                filtered_neighbors_idx = neighbors_idx[sorted_index[:num_kept_neighbors]]
            elif model_name in ["DoubleGCN_", "PEGCN", "ParaPEGCN",  "TGCN", "HelmholtzGCN", "THelmholtzGCN", "PFGCN",
                                "TPFGCN", "PIGCN", "PIGCA", "PhysEvent", "PhysEvent2", "GWNs",
                                'DoubleGCN_7N', 'GWN_7N', 'HelmholtzGCN_7N', 'PFGCN_7N', 'PIGCN_7N',]:
                num_kept_neighbors = 7
                filtered_neighbors_idx = neighbors_idx[sorted_index[:num_kept_neighbors]]  # 注意这里的sort_index指的是最佳neighbor在neighbors_idx中的索引，草了！
            else:
                num_kept_neighbors = 5
                random_index = torch.randperm(num_neighbors)[:num_kept_neighbors]
                filtered_neighbors_idx = neighbors_idx[random_index]
            remain_node_index = torch.cat((remain_node_index, filtered_neighbors_idx))

        remain_node_index = remain_node_index.type('torch.LongTensor')
        edge_index = edge_index[:, remain_node_index]
        multi_remain_data.append(edge_index)

    return multi_remain_data
