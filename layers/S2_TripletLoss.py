# -*- coding: utf-8 -*-

from itertools import combinations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Applies an average on seq, of shape(nodes, features)
class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()
    def forward(self, seq):
        return torch.mean(seq, 0)


class OnlineTripletLoss(nn.Module):
    '''
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels
    Triplets are generated using triplet_selector objects that take embeddings and targets and return indices of triplets
    '''

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):  # (100, 192); target, (100, )
        triplets = self.triplet_selector.get_triplets(embeddings, target)
        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)
        losses = F.relu(ap_distances - an_distances + self.margin)  # (179, )

        return losses.mean(), len(triplets)

class TripletSelector:
    '''
    Implementation should return indices of anchors, positive and negative samples
    return np array of shape [N_triplets * 3]
    '''
    def __init__(self):
        pass
    def get_triplets(self, embeddings, labels):
        raise NotImplementedError


def distance_matrix_computation(vectors):
    distance_matrix=-2*vectors.mm(torch.t(vectors))+vectors.pow(2).sum(dim=1).view(1,-1)+vectors.pow(2).sum(dim=1).view(-1,1)
    return distance_matrix


class FunctionNegativeTripletSelector(TripletSelector):
    '''
    For each positive pair, takes the hardest negative sample (with the greatest triplet loss value) to create a triplet
    Margin should match the margin used in triplet loss.
    negative_selection_fn should take array of loss_values for a given anchor-positive pair and all negative samples
    and return a negative index for that pair
    '''

    def __init__(self, margin, negative_selection_fn, cpu=False):
        super(FunctionNegativeTripletSelector, self).__init__()
        self.cpu = cpu
        self.margin = margin
        self.negative_selection_fn = negative_selection_fn  # 返回loss_values最大元素值的index的selector

    def get_triplets(self, embeddings, labels):  # (100, 192); target, (100, )
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = distance_matrix_computation(embeddings)  # pre embedding计算distance matrix (100, 100)
        distance_matrix = distance_matrix

        triplets = []

        for label in set(labels):
            label_mask = (labels == label)  # numpy array, (100,), ([True, False, True, True])
            label_indices = torch.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = torch.where(torch.logical_not(label_mask))[0]  # not_label_index, array([1], dtype=int64)
            anchor_pos_list = list(combinations(label_indices, 2))  #  list: 3, [(23, 66), (23, 79), (66, 79)]
            anchor_pos_list = torch.asarray(anchor_pos_list)

            anchor_p_distances = distance_matrix[anchor_pos_list[:, 0], anchor_pos_list[:, 1]]  # (3, ),tensor([-1.1761,-0.8381,0.0099])
            for anchor_positive, ap_distance in zip(anchor_pos_list, anchor_p_distances):
                loss_values = ap_distance - distance_matrix[torch.LongTensor(torch.asarray([anchor_positive[0]])), torch.LongTensor(negative_indices)] + self.margin  # anchor_positive, (2, ) [23, 66];
                hard_neg_max_index = self.negative_selection_fn(loss_values)
                if hard_neg_max_index is not None:
                    hard_negative = negative_indices[hard_neg_max_index]
                    triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])  # positive label

        if len(triplets) == 0:
            print('triplets warning!!')
            print(label)
            print(set(labels))
            triplets.append([anchor_positive[0], anchor_positive[1], negative_indices[0]])
            print(triplets)
        triplets = torch.asarray(triplets)  # (179, 3)
        return torch.LongTensor(triplets)

class FunctionNPairLoss(nn.Module):

    def __init__(self, margin, cpu=True):
        super(FunctionNPairLoss, self).__init__()
        self.cpu = cpu
        self.margin = margin

    def forward(self, embeddings, labels):  # (100, 192); target, (100, )
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = distance_matrix_computation(embeddings)  # pre embedding计算distance matrix (100, 100)
        distance_matrix = distance_matrix.cpu()

        labels = labels.cpu().data.numpy()  # 100

        n_pair_loss_list = []
        for label in set(labels):
            label_mask = (labels == label)  # numpy array, (100,), ([True, False, True, True])
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_pos_list = list(combinations(label_indices, 2))
            anchor_pos_list = np.array(anchor_pos_list)

            anchor_p_distances = distance_matrix[
                anchor_pos_list[:, 0], anchor_pos_list[:, 1]]
            loss_list = []
            for anchor_positive, ap_distance in zip(anchor_pos_list, anchor_p_distances):
                loss_values = distance_matrix[  # loss_values, (97, );
                    torch.LongTensor(np.array([anchor_positive[0]])), torch.LongTensor(negative_indices)] - ap_distance  # anchor_positive, (2, ) [23, 66]
                loss_value = torch.exp(loss_values).sum() + 1  # (97, )
                loss_list.append(torch.log(loss_value))
            n_pair_loss_list.append(torch.FloatTensor(loss_list).mean())
        n_pair_loss_value = torch.FloatTensor(n_pair_loss_list).mean()
        return n_pair_loss_value.requires_grad_(True), len(n_pair_loss_list)

def random_hard_negative(loss_values):
    hard_negatives = np.where(loss_values > 0)[0]
    return np.random.choice(hard_negatives) if len(hard_negatives) > 0 else None

def hardest_negative(loss_values):
    hard_negative = torch.argmax(loss_values)
    return hard_negative if loss_values[hard_negative] > 0 else None

def HardestNegativeTripletSelector(margin, cpu=False):
    return FunctionNegativeTripletSelector(margin=margin, negative_selection_fn=hardest_negative, cpu=cpu)

def RandomNegativeTripletSelector(margin, cpu=False):
    return FunctionNegativeTripletSelector(margin=margin, negative_selection_fn=random_hard_negative, cpu=cpu)