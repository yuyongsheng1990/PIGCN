# -*- coding: utf-8 -*-

import random
import numpy as np
import json
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import gc
import time
from typing import List

import os
project_path = os.getcwd()

from layers.S2_TripletLoss import OnlineTripletLoss, HardestNegativeTripletSelector, RandomNegativeTripletSelector
from layers.S3_NeighborRL import cal_similarity_node_edge, RL_neighbor_filter


from models.HelmholtzGCN import HelmholtzGCN
from models.PFGCN import PFGCN
from models.PIGCN import PIGCN

from utils.S2_gen_dataset import create_offline_homodataset, create_multi_relational_graph, create_homograph_relational, MySampler, save_embeddings
from utils.S4_Evaluation import AverageNonzeroTripletsMetric, evaluate
from layers import aug, discriminator

from baselines.MarGNN import MarGNN
from baselines.GraphFormer import GraphFormer
from baselines.ReDHAN import ReDHAN
from baselines.LDA import LDA
from baselines.SBERT import SBERT

from baselines.DoubleGCN import DoubleGCN
from baselines.GWN import GWN

# Helmholtz Optimizers
from torch_optimizer import Adahessian
from ranger_adabelief import RangerAdaBelief

def args_register():
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_epochs', default=50, type=int, help='Number of initial-training/maintenance-training epochs.')
    parser.add_argument('--window_size', default=3, type=int, help='Maintain the model after predicting window_size blocks.')
    parser.add_argument('--patience', default=5, type=int,
                        help='Early stop if perfermance did not improve in the last patience epochs.')
    parser.add_argument('--margin', default=8, type=float, help='Margin for computing triplet losses')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--batch_size', default=100, type=int,
                        help='Batch size (number of nodes sampled to compute triplet loss in each batch)')
    parser.add_argument('--hid_dim', default=256, type=int, help='Hidden dimension')
    parser.add_argument('--out_dim', default=256, type=int, help='Output dimension of tweet representation')
    parser.add_argument('--heads', default=8, type=int, help='Number of heads used in GAT')
    parser.add_argument('--validation_percent', default=0.2, type=float, help='Percentage of validation nodes(tweets)')
    parser.add_argument('--attn_drop', default=0.5, type=float, help='masked probability for attention layer')
    parser.add_argument('--feat_drop', default=0.0, type=float, help='dropout probability for feature embedding in attn')
    parser.add_argument('--use_hardest_neg', dest='use_hardest_neg', default=False, action='store_true',
                        help='If true, use hardest negative messages to form triplets. Otherwise use random ones')
    parser.add_argument('--is_shared', default=False)
    parser.add_argument('--inter_opt', default='cat_w_avg')
    parser.add_argument('--is_initial', default=True)
    parser.add_argument('--sampler', default='RL_sampler')
    parser.add_argument('--cluster_type', default='kmeans', help='Types of clustering algorithms')  # DBSCAN
    parser.add_argument('--time_lambda', default=0.2, type=float, help='The hyperparameter of time exponential decay')  # DHAN 时间衰减参数 lambda
    parser.add_argument('--gl_eps', default=2, type=float, help='the temperature param for global-local GCL function')

    # RL-0，learn the optimal neighbor weights
    parser.add_argument('--threshold_start0', default=[[0.2], [0.2], [0.2]], type=float,
                        help='The initial value of the filter threshold for state1 or state3')
    parser.add_argument('--RL_step0', default=0.02, type=float, help='The starting epoch of RL for state1 or state3')
    parser.add_argument('--RL_start0', default=0, type=int, help='The starting epoch of RL for state1 or state3')

    # RL-1，learn the optimal DBSCAN params.
    parser.add_argument('--eps_start', default=0.001, type=float, help='The initial value of the eps for state2')
    parser.add_argument('--eps_step', default=0.02, type=float, help='The step size of eps for state2')
    parser.add_argument('--min_Pts_start', default=2, type=int, help='The initial value of the min_Pts for state2')
    parser.add_argument('--min_Pts_step', default=1, type=int, help='The step size of min_Pts for state2')

    # other arguments
    parser.add_argument('--use_cuda', dest='use_cuda', default=True, action='store_true', help='Use cuda')
    parser.add_argument('--data_path', default=project_path + '/data', type=str, help='graph data path')

    parser.add_argument('--mask_path', default=None, type=str,
                        help='File path that contains the training, validation and test masks')
    parser.add_argument('--log_interval', default=10, type=int, help='Log interval')

    args = parser.parse_args(args=[])

    return args

# # tensor version: 将二维矩阵list 转换成adj matrix list
def relations_to_adj(r_data, nb_nodes=None, device=None):
    data = torch.ones(r_data.shape[1]).to(device)
    relation_mx = torch.sparse_coo_tensor(indices=r_data, values=data, size=[nb_nodes, nb_nodes],
                                          dtype=torch.int32)
    return relation_mx.to_dense()

# tensor version: 计算偏差矩阵
def adj_to_bias(adj, nhood=1, device=None):  # adj,(3025, 3025); sizes, [3025]
    mt = torch.eye(adj.shape[0]).to(device)
    for _ in range(nhood):
        adj = torch.add(adj, torch.eye(adj.shape[1]).to(device))
        mt = torch.matmul(mt, adj)  # 相乘
    mt = torch.where(mt > 0, 1, mt)
    return (-1e9 * (1.0 - mt))  # 科学计数法，2.5 x 10^(-27)表示为：2.5e-27

def offline_FinEvent_model(i,  # message block_i=0
                           dataset_name,
                           text_vectors,
                           model_name,
                           args,
                           metrics,
                           data_path,  # args.data_path + '/{}_offline_embeddings/'
                           loss_fn,
                           optimizer_name,
                           model=None):

    start_running_time = time.time()
    # step1: make dir for graph i
    embeddings_path = data_path + f'block_{i}/{i}_{text_vectors}'
    if not os.path.exists(embeddings_path):
        os.makedirs(embeddings_path)
    save_path_i = data_path + f'block_{i}/{model_name}_{text_vectors}'
    if not os.path.exists(save_path_i):
        os.makedirs(save_path_i)

    # CUDA
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu')

    # step2: load datal
    relation_ids: List[str] = ['entity', 'userid', 'word']  # twitter dataset
    homo_data = create_offline_homodataset(embeddings_path)  # (4762, 302): feature embedding; y: label, generate train_slices (3334), val_slices (952), test_slices (476)
    multi_r_data = create_multi_relational_graph(embeddings_path, relation_ids)
    num_relations = len(multi_r_data)  # 3

    # data.train_mask, data.val_mask, data.test_mask = gen_offline_masks(len(labels))
    train_num_samples, valid_num_samples, test_num_samples = homo_data.train_mask.size(0), homo_data.val_mask.size(
        0), homo_data.test_mask.size(0)  # 3354, 952, 476
    all_num_samples = train_num_samples + valid_num_samples + test_num_samples
    torch.save(homo_data.train_mask, save_path_i + '/train_mask.pt')
    torch.save(homo_data.val_mask, save_path_i + '/valid_mask.pt')
    torch.save(homo_data.test_mask, save_path_i + '/test_mask.pt')

    # input dimension (300 in our paper)
    num_dim = homo_data.x.size(0)  # 4762
    feat_dim = homo_data.x.size(1)  # embedding dimension, 302
    nb_classes = len(np.unique(homo_data.y))

    # prepare graph configs for node filtering
    if args.is_initial:
        print('prepare node configures...')
        cal_similarity_node_edge(multi_r_data, homo_data.x, save_path_i)
        filter_path = save_path_i
    else: filter_path = None
    prepare_time = (time.time() - start_running_time) / 60
    # logging
    message = '\n{} model detect social events on {} vectors on {} dataset. \n'.format(model_name, text_vectors, dataset_name)
    message += f'detection model prepare graph data time from start: {prepare_time:.4f} mins. \n'

    # sampling size
    if model_name == 'ReDHAN':
        sample_size = [-1]
    else:
        sample_size = [-1, -1]
    RL_thresholds = torch.FloatTensor(args.threshold_start0).to(device)  # [[0.2], [0.2], [0.2]]

    # RL_filter means extract limited sorted neighbors based on RL_threshold and neighbor similarity, return filtered node -> neighbor index
    if args.sampler == 'RL_sampler':
        filtered_multi_r_data = RL_neighbor_filter(multi_r_data, RL_thresholds, filter_path, model_name)  # filtered (2,104479); (2,6401); (2,15072)
    else:
        filtered_multi_r_data = multi_r_data

    message += 'event classes: ' + str(nb_classes) + '\n'
    message += ''.join('message numbers: ' + str(num_dim)) + '\n'
    message += ''.join(['edge numbers: '+ str(multi_r_data[j].shape) for j in range(num_relations)]) + '\n'
    message += ''.join(['filtered message numbers: '+ str(filtered_multi_r_data[j].shape) for j in range(num_relations)]) + '\n'
    print(message)
    with open(save_path_i + '/log.txt', 'a') as f:
        f.write(message)

    print('Pre-Train Stage...')
    if model_name == "ReDHAN":
        # ReDHAN model with RL_filter and Neighbor_sampler，这要torch_geometric重写HAN模型，要不然用不上FinEvent中的neighbor_sampler.
        # 所以，既要用到adjs_list for RL_sampler，也要用到bias_list for HAN algorithm.
        adj_mx_list = [relations_to_adj(filtered_r_data, torch.tensor(num_dim).to(device), device) for filtered_r_data in filtered_multi_r_data]  # 邻接矩阵list:3,tensor, (4762,4762)
        biases_mat_list = [adj_to_bias(adj, torch.tensor(1).to(device), device) for adj in adj_mx_list]  # 偏差矩阵list:3,tensor, (4762,4762)
        model = ReDHAN(feature_size=feat_dim, nb_classes=nb_classes, nb_nodes=num_dim, attn_drop=args.attn_drop,
                                        feat_drop=args.feat_drop, hid_dim=args.hid_dim, out_dim=args.out_dim, time_lambda=args.time_lambda,  # 时间衰减参数，默认: -0.2
                                        num_relations=num_relations, hid_units=[8], n_heads=[8,1])
    elif model_name in ['DoubleGCN', 'DoubleGCN_3N', 'DoubleGCN_7N']:
        # baseline 3: DoubleGCN
        model = DoubleGCN(feature_size=feat_dim, hid_dim=args.hid_dim, out_dim=args.out_dim, num_relations=num_relations)
    elif model_name == 'GraphFormer': # baseline 2: GraphFormer.
        model = GraphFormer(feature_size=feat_dim, hid_dim=args.hid_dim, out_dim=args.out_dim, num_relations=num_relations)
    elif model_name == 'FinEvent':
        # baseline 1: FinEvent, feat_dim=302; hidden_dim=128; out_dim=64; heads=4; inter_opt=cat_w_avg; is_shared=False
        model = MarGNN((feat_dim, args.hid_dim, args.out_dim, args.heads),
                       num_relations=num_relations, inter_opt=args.inter_opt, is_shared=args.is_shared)
    elif model_name in ['HelmholtzGCN', 'HelmholtzGCN_3N', 'HelmholtzGCN_7N']:
        model = HelmholtzGCN(feature_size=feat_dim, hid_dim=args.hid_dim, out_dim=args.out_dim, num_relations=num_relations)
    elif model_name in ['PFGCN', 'PFGCN_3N', 'PFGCN_7N']:
        model = PFGCN(feature_size=feat_dim, hid_dim=args.hid_dim, out_dim=args.out_dim, num_relations=num_relations)
    elif model_name in ['PIGCN', 'PIGCN_3N', 'PIGCN_7N', 'PIGCA']:
        model = PIGCN(feature_size=feat_dim, hid_dim=args.hid_dim, out_dim=args.out_dim, num_relations=num_relations)
    elif model_name in ['GWN', 'GWN_3N', 'GWN_7N']:
        model = GWN(feature_size=feat_dim, hid_dim=args.hid_dim, num_relations=num_relations)
    elif model_name == 'SBERT':
        # baseline 3: SBERT embedding
        sbert_embeddings = SBERT(dataset_name, i)
        sbert_embeddings = torch.FloatTensor(sbert_embeddings)
        print(sbert_embeddings.shape)
        print('-------------------SBERT----------------------------')
        save_embeddings(sbert_embeddings, save_path_i, file_name='final_embeddings.pt')
        save_embeddings(homo_data.y, save_path_i, file_name='labels.pt')
        test_nmi = evaluate(sbert_embeddings[homo_data.test_mask],
                            homo_data.y,
                            indices=homo_data.test_mask,
                            epoch=-1 if i == 0 else -2,
                            num_isolated_nodes=0,
                            save_path=save_path_i,
                            is_validation=False,
                            cluster_type=args.cluster_type)
        # save model
        model_path = save_path_i + '/models'  # data_path + 'block_{}/{}_{}'.format(i, text_vectors, model_name)
        if not os.path.exists(model_path): os.mkdir(model_path)
        mins_spent = (time.time() - start_running_time) / 60
        message += '\nWhole Running Time took {:.2f} mins'.format(mins_spent)
        message += '\n'
        print(message)
        with open(save_path_i + '/log.txt', 'a') as f:
            f.write(message)
        return test_nmi, None
    elif model_name == 'LDA':
        # baseline 4: LDA embedding
        lda_embeddings = LDA(dataset_name, i)
        lda_embeddings = torch.FloatTensor(lda_embeddings)
        print(lda_embeddings.shape)
        print('-------------------LDA----------------------------')
        save_embeddings(lda_embeddings, save_path_i, file_name='final_embeddings.pt')
        save_embeddings(homo_data.y, save_path_i, file_name='labels.pt')
        test_nmi = evaluate(lda_embeddings[homo_data.test_mask],
                            homo_data.y,
                            indices=homo_data.test_mask,
                            epoch=-1 if i == 0 else -2,
                            num_isolated_nodes=0,
                            save_path=save_path_i,
                            is_validation=False,
                            cluster_type=args.cluster_type)
        # save model
        model_path = save_path_i + '/models'  # data_path + 'block_{}/{}_{}'.format(i, text_vectors, model_name)
        if not os.path.exists(model_path): os.mkdir(model_path)
        mins_spent = (time.time() - start_running_time) / 60
        message += '\nWhole Running Time took {:.2f} mins'.format(mins_spent)
        message += '\n'
        print(message)
        with open(save_path_i + '/log.txt', 'a') as f:
            f.write(message)
        return test_nmi, None
    else: biases_mat_list = None

    # define sampler
    sampler = MySampler(args.sampler)  # RL_sampler
    # load model to device
    model.to(device)

    # define optimizer
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    elif optimizer_name == 'AdaHessian':
        optimizer = Adahessian(model.parameters(), lr=args.lr, weight_decay=1e-4)
    elif optimizer_name == 'Ranger':
        optimizer = RangerAdaBelief(model.parameters(), lr=args.lr, weight_decay=1e-4)
    else:
        raise ValueError(f'Unsupproted optimizer: {optimizer_name}')
    # step12.0: record the highest validation nmi ever got for early stopping
    best_vali_nmi = 1e-9
    best_epoch = 0
    wait = 0
    gcl_loss_fn = nn.BCEWithLogitsLoss()
    gcl_disc = discriminator.Discriminator(args.out_dim)
    gcl_disc.to(device)

    if i == 0:
        message = f'\n------------------Start detection on initial block_{str(i)}------------------------\n'
        with open(save_path_i + '/log.txt', 'a') as f:
            f.write(message)
    else:
        message = f'\n------------------Incremental Detecting on block_{str(i)}--------------------------\n'
        model.eval()
        message += f"\n-----------------loading best model in the previous block_{str(i - 1)}---------------\n"
        # step18: args.data_path + '/{}_offline_embeddings/'
        best_model_path = data_path + 'block_{}/{}_{}/models/best_model.pt'.format(str(i - 1), model_name, text_vectors)
        checckpoint = torch.load(best_model_path)
        model.load_state_dict(checckpoint['model'], strict=False)
        print('Last Stream best model loaded.')

        # we recommend to forward all nodes and select the validation indices instead
        extract_features = torch.FloatTensor([])
        num_batches = int(all_num_samples / args.batch_size) + 1  # 这里的all_num_samples,是为了然后epoch n对应的model求feature,而不是用于evaluation

        # all mask are then splited into mini-batch in order
        all_mask = torch.arange(0, num_dim, dtype=torch.long)

        for batch in range(num_batches):
            # split batch
            i_start = args.batch_size * batch
            i_end = (batch + 1) * args.batch_size
            batch_nodes = all_mask[i_start:i_end]

            # sampling neighbors of batch nodes；ReDHAN sample_size=[-1], 选取所有的邻居
            adjs, n_ids, sampled_multi_r_data = sampler.sample(filtered_multi_r_data, node_idx=batch_nodes,
                                                               sizes=sample_size, batch_size=args.batch_size)
            if model_name == 'ReDHAN':
                pred = model(homo_data.x, biases_mat_list, batch_nodes, adjs, n_ids, device, RL_thresholds)  # ReDHAN model
            if model_name == "GraphFormer":
                pred = model(homo_data.x, batch_nodes, adjs, n_ids, device)  # GraphFormer model
            if model_name == 'FinEvent':
                pred = model(homo_data.x, adjs, n_ids, device, RL_thresholds)  # Fin-Event pred: embedding (100,192)
            if model_name in ['DoubleGCN', 'DoubleGCN_3N', 'DoubleGCN_7N']:
                pred = model(homo_data.x, filtered_multi_r_data, batch_nodes, device)  # DoubleGCN
            if model_name in ["HelmholtzGCN", 'HelmholtzGCN_3N', 'HelmholtzGCN_7N', "THelmholtzGCN"]:
                # pred = model(homo_data.x, filtered_multi_r_data, batch_nodes, device)  # HelmholtzGCN
                pred, _ = model(homo_data.x, filtered_multi_r_data, batch_nodes, device)  # HelmholtzGCN
            if model_name in ['PFGCN', 'PFGCN_3N', 'PFGCN_7N', "TPFGCN"]:
                pred = model(homo_data.x, filtered_multi_r_data, batch_nodes, device)  # PFGCN
            if model_name in ['PIGCN', 'PIGCN_3N', 'PIGCN_7N', 'PIGCA']:
                pred, _ = model(homo_data.x, filtered_multi_r_data, batch_nodes, device)  # PIGCN
                # pred, cos_loss, helm_loss = model(homo_data.x, filtered_multi_r_data, batch_nodes, device)  # PIGCN
            if model_name in ['GWN', 'GWN_3N', 'GWN_7N']:
                pred = model(homo_data.x, filtered_multi_r_data, batch_nodes, device)  # GWN

            extract_features = torch.cat((extract_features, pred.detach()), dim=0)
            del pred
            gc.collect()

        # data_path + 'block_{}/{}_{}'.format(i, text_vectors, model_name)
        save_embeddings(extract_features, save_path_i, file_name='incre_embeddings.pt')

        former_save_path = data_path + f'/block_{str(i - 1)}/{model_name}_{text_vectors}'
        detection_nmi = evaluate(extract_features[all_mask],
                                 homo_data.y,
                                 indices=all_mask,
                                 epoch=-2,
                                 num_isolated_nodes=0,
                                 save_path=save_path_i,
                                 former_save_path=former_save_path,
                                 is_validation=False,
                                 cluster_type=args.cluster_type)  # 'dbscan'

        detection_mins = (time.time() - start_running_time) / 60
        message += f'\nIncremental Detection Time took {detection_mins:.2f} mins from start.\n'
        print(message)
        with open(save_path_i + '/log.txt', 'a') as f:
            f.write(message)

    # step13: start training------------------------------------------------------------
    if i==0:
        message = '\n------------------start training model------------------------\n'
    else:
        message = '\n------------------maintaining model------------------------\n'
    print(message)
    with open(save_path_i + '/log.txt', 'a') as f:
        f.write(message)
    # step12.1: record validation nmi of all epochs before early stop
    all_vali_nmi = []
    # step12.2: record the time spent in seconds on each batch of all training/maintaining epochs
    train_batch_mins = []
    # step12.3: record the time spent in mins on each epoch
    train_epoch_mins = []
    gcl_loss = None
    cos_loss = None  # for alignment of physical force and helm_loss.
    helm_loss = None
    for epoch in range(args.n_epochs):
        start_epoch_time = time.time()
        losses = []
        total_loss = 0.0

        for metric in metrics:
            metric.reset()

        # step13.0: forward
        model.train()

        # mini-batch training
        batch = 0
        num_batches = int(train_num_samples / args.batch_size)  # 34
        for batch in range(num_batches):
            start_batch_time = time.time()
            # split batch
            i_start = args.batch_size * batch
            i_end = min((batch + 1) * args.batch_size, train_num_samples)
            batch_nodes = homo_data.train_mask[i_start:i_end]  # 100个train_idx
            batch_labels = homo_data.y[batch_nodes].to(device)
            if torch.unique(batch_labels).size(0) < 2:
                continue

            # sampling neighbors of batch nodes
            adjs, n_ids, sampled_multi_r_data = sampler.sample(filtered_multi_r_data, node_idx=batch_nodes, sizes=sample_size,
                                         batch_size=args.batch_size)
            optimizer.zero_grad()

            if model_name == 'ReDHAN':
                pred = model(homo_data.x, biases_mat_list, batch_nodes, adjs, n_ids, device, RL_thresholds)  # ReDHAN model
            if model_name == "GraphFormer":
                pred = model(homo_data.x, batch_nodes, adjs, n_ids, device)  # GraphFormer model
            if model_name in ['FinEvent', 'TGAT']:
                pred = model(homo_data.x, adjs, n_ids, device, RL_thresholds)  # Fin-Event pred: embedding (100,192)
            if model_name in ["DoubleGCN", 'DoubleGCN_3N', 'DoubleGCN_7N', "TGCN", 'ParaPEGCN', 'PEGCN']:
                pred = model(homo_data.x, filtered_multi_r_data, batch_nodes, device)  # DoubleGCN
            if model_name in ["HelmholtzGCN", 'HelmholtzGCN_3N', 'HelmholtzGCN_7N', "THelmholtzGCN"]:
                # pred = model(homo_data.x, filtered_multi_r_data, batch_nodes, device)  # HelmholtzGCN
                pred, helm_loss = model(homo_data.x, filtered_multi_r_data, batch_nodes, device)  # HelmholtzGCN
            if model_name in ['PFGCN', 'PFGCN_3N', 'PFGCN_7N', "TPFGCN"]:
                pred = model(homo_data.x, filtered_multi_r_data, batch_nodes, device)  # PFGCN
            if model_name in ['PIGCN', 'PIGCN_3N', 'PIGCN_7N', 'PIGCA']:
                pred, helm_loss = model(homo_data.x, filtered_multi_r_data, batch_nodes, device)  # PIGCN
                # pred, cos_loss, helm_loss = model(homo_data.x, filtered_multi_r_data, batch_nodes, device)  # PIGCN
            if model_name in ['GWN', 'GWN_3N', 'GWN_7N']:
                pred = model(homo_data.x, filtered_multi_r_data, batch_nodes, device)  # GWN

            '''----------triplet_loss--------------'''
            loss_outputs = loss_fn(pred, batch_labels)  # (12.8063), 179
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs  # GCN loss
            print('triplet_loss: ', loss)

            if model_name == 'ReDHAN':
                '''----------new GraphCL v1.0 with disc_cosine: comparison of batch graph with subgraph augmentation ------------------------'''
                # Random sample 80% nodes from batch_features
                random_idx_1 = torch.LongTensor(
                    random.sample(range(batch_nodes.shape[0]), int(batch_nodes.shape[0] * 0.8)))  # 0.8; 0.9
                random_idx_2 = torch.LongTensor(
                    random.sample(range(batch_nodes.shape[0]), int(batch_nodes.shape[0] * 0.9)))  # 0.8; 0.9
                sub_nodes_1 = torch.index_select(batch_nodes, 0, random_idx_1).to(device)  # 采样维度 dim=0
                sub_nodes_2 = torch.index_select(batch_nodes, 0, random_idx_2).to(device)  # 采样维度 dim=0
                # subgraph feature embeddings
                # 归一化
                gcl_normalized_r_data_list = [aug.normalize_adj(torch.add(adj, torch.eye(adj.shape[0]).to(device))) for adj in biases_mat_list]  # 原始adj matrix做归一化normalize, ndarray, (3327,3327)
                # negative samples
                features_neg = homo_data.x.clone()
                features_neg = features_neg[torch.randperm(features_neg.shape[0])]
                # 构建标签label. Bilinear的值域为[0,1] 或[-1, 1], 值域变化受输入数据影响
                lbl_1 = torch.ones(batch_nodes.shape[0], 1)  # labels for aug_1, (1,192)
                lbl_2 = torch.zeros(batch_nodes.shape[0], 1)  # (1,192)
                lbl = torch.cat((lbl_1, lbl_2), dim=0)  # (1,128)
                # 基于data augmentation生成关于original features和shuffled features的embedding
                h_pos = pred.clone()
                h_neg = model(features_neg, gcl_normalized_r_data_list, batch_nodes, adjs, n_ids, device, RL_thresholds)  # HAN_2构建负样本 negative feature embeddings
                # h_neg = model(features_neg, adjs, n_ids, device, RL_thresholds)   # FinEvent
                # 构建subgraph augmentation embedding
                aug_adjs_1, aug_n_ids_1, _ = sampler.sample(filtered_multi_r_data,
                                                         node_idx=sub_nodes_1, sizes=sample_size,
                                                         batch_size=len(sub_nodes_1))  # RL_sampler from aug_adj for HAN_2
                h_aug_1 = model(homo_data.x, biases_mat_list, sub_nodes_1, aug_adjs_1, aug_n_ids_1, device, RL_thresholds)  # ReDHAN 构建 subgraph augmentation embeddings, (90, 64)
                aug_adjs_2, aug_n_ids_2, _ = sampler.sample(filtered_multi_r_data,
                                                         node_idx=sub_nodes_2, sizes=sample_size,
                                                         batch_size=len(sub_nodes_2))  # RL_sampler from aug_adj for HAN_2
                h_aug_2 = model(homo_data.x, biases_mat_list, sub_nodes_2, aug_adjs_2, aug_n_ids_2, device, RL_thresholds)  # ReDHAN 计算正样本 subgraph augmentation embeddings
                # discriminator. Bilinear双向线性映射，将subgraph embedding 与pos embedding对齐；将sub embedding 2与neg embedding对齐。pos对齐，相似度为1，neg为0.
                # logits, (1,6654)
                # readout
                c_aug_1 = torch.sigmoid(torch.mean(h_aug_1, 0))  # (64,)
                c_aug_2 = torch.sigmoid(torch.mean(h_aug_2, 0))
                ret_1 = gcl_disc(c_aug_1, h_pos, h_neg, device)  # 鉴别器，本质上是一个预估的插值，做平滑smooth用，它可以对输入图像的微小变化具有一定的鲁棒性
                ret_2 = gcl_disc(c_aug_2, h_pos, h_neg, device)  # (100, 384) # BiLinear
                ret = ret_1 + ret_2
                gcl_loss = gcl_loss_fn(ret.cpu(), lbl)  # ret, (1,128); lbl, (1,128)
                print('gcl_loss: ', gcl_loss)
                message = 'gcl_loss: {:.2f}.'.format(gcl_loss)
                with open(save_path_i + '/log.txt', 'a') as f:
                    f.write(message)

            if model_name == 'ReDHAN':
                loss = loss + gcl_loss  # 0.823; 0.732; 0.657
            # helmholtzGCN loss
            if helm_loss is not None:
                loss = loss +  0.1 * helm_loss  # random Gaussian noise ε to prevent over-fitting.
            if cos_loss is not None:
                loss = loss + cos_loss

            losses.append(loss.item())
            total_loss += loss.item()

            # step13.1: metrics
            for metric in metrics:
                metric(pred, batch_labels, loss_outputs)
            if batch % args.log_interval == 0:
                message = 'Train: [{}/{} ({:.0f}%)] \tloss: {:.6f}'.format(batch * args.batch_size, train_num_samples,
                                                                           100. * batch / ((train_num_samples // args.batch_size) + 1),
                                                                           np.mean(losses))
                for metric in metrics:
                    message += '\t{}: {:.4f}'.format(metric.name(), metric.value())  # Average nonzero triplets: 1149.0000
                # print(message)
                with open(save_path_i + '/log.txt', 'a') as f:
                    f.write(message)
                losses = []

            del pred, loss_outputs
            gc.collect()

            # step13.2: backward
            if optimizer_name == 'AdaHessian':
                loss.backward(create_graph=True)
            else:
                loss.backward()     # Adam; Ranger

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            batch_time_mins = (time.time() - start_batch_time) / 60
            train_batch_mins.append(batch_time_mins)

            # del loss
            gc.collect()

        # step14: print loss
        total_loss /= (batch + 1)
        message = 'Epoch: {}/{}. Average loss: {:.4f}'.format(epoch, args.n_epochs, total_loss)

        for metric in metrics:
            message += '\t{}: {:.4f}'.format(metric.name(), metric.value())
        epoch_time = (time.time() - start_epoch_time) / 60
        message += f"\nThis epoch took {epoch_time:.4f} mins. \n"
        print(message)
        with open(save_path_i + '/log.txt', 'a') as f:
            f.write(message)
        train_epoch_mins.append(epoch_time)

        # step15: validation--------------------------------------------------------
        print('---------------------validation-------------------------------------')
        # inder the representations of all tweets
        model.eval()

        # we recommend to forward all nodes and select the validation indices instead
        extract_features = torch.FloatTensor([])
        num_batches = int(all_num_samples / args.batch_size) + 1

        # all mask are then splited into mini-batch in order
        all_mask = torch.arange(0, num_dim, dtype=torch.long)

        for batch in range(num_batches):
            # split batch
            i_start = args.batch_size * batch
            i_end = min((batch + 1) * args.batch_size, all_num_samples)
            batch_nodes = all_mask[i_start:i_end]

            # sampling neighbors of batch nodes
            adjs, n_ids, sampled_multi_r_data = sampler.sample(filtered_multi_r_data, node_idx=batch_nodes, sizes=sample_size,
                                         batch_size=args.batch_size)

            if model_name == 'ReDHAN':
                pred = model(homo_data.x, biases_mat_list, batch_nodes, adjs, n_ids, device, RL_thresholds)  # ReDHAN model
            if model_name in ['FinEvent', "TGAT"]:
                pred = model(homo_data.x, adjs, n_ids, device, RL_thresholds)  # embedding (100,192)
            if model_name == "GraphFormer":
                pred = model(homo_data.x, batch_nodes, adjs, n_ids, device)  # GraphFormer model
            if model_name in ["DoubleGCN", 'DoubleGCN_3N', 'DoubleGCN_7N', "TGCN", "ParaPEGCN", "PEGCN"]:
                pred = model(homo_data.x, filtered_multi_r_data, batch_nodes, device)  # DoubleGCN
            if model_name in ["HelmholtzGCN", 'HelmholtzGCN_3N', 'HelmholtzGCN_7N', "THelmholtzGCN"]:
                # pred = model(homo_data.x, filtered_multi_r_data, batch_nodes, device)  # HelmholtzGCN
                pred, _ = model(homo_data.x, filtered_multi_r_data, batch_nodes, device)  # HelmholtzGCN
            if model_name in ["PFGCN", 'PFGCN_3N', 'PFGCN_7N', "TPFGCN"]:
                pred = model(homo_data.x, filtered_multi_r_data, batch_nodes, device)  # PFGCN
            if model_name in ['PIGCN', 'PIGCN_3N', 'PIGCN_7N', 'PIGCA']:
                pred, _ = model(homo_data.x, filtered_multi_r_data, batch_nodes, device)  # PIGCN
                # pred, _, _ = model(homo_data.x, filtered_multi_r_data, batch_nodes, device)  # PIGCN
            if model_name in ['GWN', 'GWN_3N', 'GWN_7N']:
                pred = model(homo_data.x, filtered_multi_r_data, batch_nodes, device)  # GWN

            extract_features = torch.cat((extract_features, pred.cpu().detach()), dim=0)

            del pred
            gc.collect()

        # evaluate the model: conduct kMeans clustering on the validation and report NMI
        validation_nmi = evaluate(extract_features[homo_data.val_mask],  # val feature embedding
                                  homo_data.y,
                                  indices=homo_data.val_mask,
                                  epoch=epoch,
                                  num_isolated_nodes=0,
                                  save_path=save_path_i,
                                  former_save_path=None,
                                  is_validation=True,
                                  cluster_type=args.cluster_type)
        all_vali_nmi.append(validation_nmi)

        message = 'Epoch: {}/{}. Validation_nmi : {:.4f}'.format(epoch, args.n_epochs, validation_nmi)
        with open(save_path_i + '/log.txt', 'a') as f:
            f.write(message)

        # step16: early stop
        if validation_nmi > best_vali_nmi:
            best_vali_nmi = validation_nmi
            best_epoch = epoch
            wait = 0
            # save model
            model_path = save_path_i + '/models'  # data_path + 'block_{}/{}_{}'.format(i, text_vectors, model_name)
            if (epoch == 0) and (not os.path.isdir(model_path)):
                os.mkdir(model_path)
            p = model_path + '/best_model.pt'
            torch.save({'epoch': epoch,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        }, p)
            print('Best model was at epoch ', str(best_epoch))
        else:
            wait += 1

        if wait >= args.patience:
            print('Saved all_mins_spent')
            print('Early stopping at epoch ', str(epoch))
            print('Best model was at epoch ', str(best_epoch))
            break
        # end one epoch

    # step17: save all validation nmi
    np.save(save_path_i + '/all_vali_nmi.npy', np.asarray(all_vali_nmi))
    # save time spent on batches
    np.save(save_path_i + '/train_batche_mins.npy', np.asarray(train_batch_mins))
    print('Saved train_batch_mins.')
    # save time spent on epochs
    np.save(save_path_i + '/train_epoch_mins.npy', np.asarray(train_epoch_mins))
    print('Saved train_epoch_mins.')


    "-----------------loading best model---------------"
    # step18: load the best model of the current block
    best_model_path = save_path_i + '/models/best_model.pt'
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model'])
    print('Best model loaded.')

    # del homo_data, multi_r_data
    torch.cuda.empty_cache()

    # test--------------------------------------------------------
    print('--------------------test----------------------------')
    model.eval()

    # we recommend to forward all nodes and select the validation indices instead
    extract_features = torch.FloatTensor([])
    num_batches = int(all_num_samples / args.batch_size) + 1

    # all mask are then splited into mini-batch in order
    all_mask = torch.arange(0, num_dim, dtype=torch.long)

    for batch in range(num_batches):
        # split batch
        i_start = args.batch_size * batch
        i_end = min((batch + 1) * args.batch_size, all_num_samples)
        batch_nodes = all_mask[i_start:i_end]

        # sampling neighbors of batch nodes
        adjs, n_ids, sampled_multi_r_data = sampler.sample(filtered_multi_r_data, node_idx=batch_nodes, sizes=sample_size,
                                         batch_size=args.batch_size)

        if model_name == 'ReDHAN':
            pred = model(homo_data.x, biases_mat_list, batch_nodes, adjs, n_ids, device, RL_thresholds)  # ReDHAN model
        if model_name in ['FinEvent', 'TGAT']:
            pred = model(homo_data.x, adjs, n_ids, device, RL_thresholds)  # Fin-Event pred: (100,192)
        if model_name == "GraphFormer":
            pred = model(homo_data.x, batch_nodes, adjs, n_ids, device)  # GraphFormer model
        if model_name in ["DoubleGCN", 'DoubleGCN_3N', 'DoubleGCN_7N', "TGCN", "ParaPEGCN", "PEGCN"]:
            pred = model(homo_data.x, filtered_multi_r_data, batch_nodes, device)  # DoubleGCN
        if model_name in ["HelmholtzGCN", 'HelmholtzGCN_3N', 'HelmholtzGCN_7N', "THelmholtzGCN"]:
            # pred = model(homo_data.x, filtered_multi_r_data, batch_nodes, device)  # HelmholtzGCN
            pred, _ = model(homo_data.x, filtered_multi_r_data, batch_nodes, device)  # HelmholtzGCN
        if model_name in ["PFGCN", 'PFGCN_3N', 'PFGCN_7N', "TPFGCN"]:
            pred = model(homo_data.x, filtered_multi_r_data, batch_nodes, device)  # PFGCN
        if model_name in ['PIGCN', 'PIGCN_3N', 'PIGCN_7N', 'PIGCA']:
            pred, _ = model(homo_data.x, filtered_multi_r_data, batch_nodes, device)  # PIGCN
            # pred, _, _ = model(homo_data.x, filtered_multi_r_data, batch_nodes, device)  # PIGCN
        if model_name in ['GWN', 'GWN_3N', 'GWN_7N']:
            pred = model(homo_data.x, filtered_multi_r_data, batch_nodes, device)  # GWN
        extract_features = torch.cat((extract_features, pred.cpu().detach()), dim=0)
        del pred
        gc.collect()


    save_embeddings(extract_features, save_path_i, file_name='final_embeddings.pt')
    save_embeddings(homo_data.y, save_path_i, file_name='final_labels.pt')

    test_nmi = evaluate(extract_features[homo_data.test_mask],
                        homo_data.y,
                        indices=homo_data.test_mask,
                        epoch=-1,
                        num_isolated_nodes=0,
                        save_path=save_path_i,
                        former_save_path=None,
                        is_validation=False,
                        cluster_type=args.cluster_type)

    mins_spent = (time.time() - start_running_time) / 60
    message += '\nWhole Running Time took {:.4f} mins from start. \n'.format(mins_spent)
    print(message)
    with open(save_path_i + '/log.txt', 'a') as f:
        f.write(message)
    return test_nmi, model.state_dict()

if __name__ == '__main__':
    # define args
    args = args_register()

    # check CUDA
    print('Using CUDA:', torch.cuda.is_available())

    # create working path
    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    print('Batch Size:', args.batch_size)
    print('Intra Agg Mode:', args.is_shared)
    print('Inter Agg Mode:', args.inter_opt)
    print('Reserve node config?', args.is_initial)

    # contrastive loss in our paper
    if args.use_hardest_neg:
        loss_fn = OnlineTripletLoss(args.margin, HardestNegativeTripletSelector(args.margin))  # margin used for computing tripletloss
    else:
        loss_fn = OnlineTripletLoss(args.margin, RandomNegativeTripletSelector(args.margin))
    # define metrics
    BCL_metrics = [AverageNonzeroTripletsMetric()]


    block_list = [0]
    dataset_list = ['Twitter']  # ['Twitter', 'CrisisLexT', 'Kawarith', 'French']
    text_list = ['SBERT_vectors'] # ['doc_vectors', 'bert_vectors', 'SBERT_vectors']
    # 'HelmholtzGCN', "THelmholtzGCN", 'PFGCN', "TPFGCN", 'PIGCN'
    model_list = ['DoubleGCN', 'PIGCN', 'GWN'] # ['LDA', 'GraphFormer', 'FinEvent', 'DoubleGCN', 'HelmholtzGCN', 'PFGCN', 'PIGCN']
    optimizer_list = ['Adam']  # 'Adam', 'AdaHessian', 'Ranger'
    for i in block_list:
        for d in dataset_list:
            dataset_name = d  # Twitter, CrisisLexT, Kawarith, French
            print(dataset_name)
            for t in text_list:
                text_vectors = t  # 'doc_vectors'
                for m in model_list:
                    model_name = m
                    for o in optimizer_list:  # change optimizer
                        optimizer_name = o
                        best_nmi = 0
                        data_path = args.data_path + '/{}_offline_embeddings/'.format(dataset_name)
                        if not os.path.exists(data_path): os.mkdir(data_path)
                        for iteration in range(5):
                            test_nmi, model_dict = offline_FinEvent_model(i=i,
                                                                          dataset_name=dataset_name,
                                                                          text_vectors=text_vectors,
                                                                          model_name=model_name,
                                                                          args=args,
                                                                          metrics=BCL_metrics,
                                                                          data_path=data_path,
                                                                          loss_fn=loss_fn,
                                                                          optimizer_name=optimizer_name,
                                                                          model=None)
                            print('test_nmi: ', test_nmi)
                            print('best_nmi: ', best_nmi)
                            if best_nmi < test_nmi:
                                print('best_model_dict change')
                                best_nmi = test_nmi
                                p = data_path + '/block_{}/{}_{}/models/best_model.pt'.format(i, model_name, text_vectors)
                                torch.save({
                                            'model':model_dict,
                                            }, p)
                        # record hyper-parameters
                        with open(data_path + '/args.txt', 'w') as f:
                            json.dump(args.__dict__, f, indent=2)
                        print('model finished')