#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  1 19:19:55 2025

@author: mdb
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, embeddings, labels):
        batch_centers = self.centers[labels]
        return ((embeddings - batch_centers) ** 2).sum(dim=1).mean()

class QuadrupletLoss(nn.Module):
    def __init__(self, 
                 margin1=0.3, 
                 margin2=0.3, 
                 num_hard_negatives=10,
                 use_hard_mining=True,    # AQUI HM 
                 use_center_loss=True,   # AQUI CENTER
                 center_loss_weight=0.01,
                 num_classes=288,
                 feat_dim=2048):
        super().__init__()
        self.margin1 = margin1
        self.margin2 = margin2
        self.num_hard_negatives = num_hard_negatives
        self.use_hard_mining = use_hard_mining
        self.use_center_loss = use_center_loss
        self.center_loss_weight = center_loss_weight

        if use_center_loss:
            assert num_classes is not None and feat_dim is not None, \
                "Para usar center loss, debes especificar num_classes y feat_dim"
            self.center_loss = CenterLoss(num_classes, feat_dim)

    def forward(self, embeddings, labels):
        device = embeddings.device
        labels = labels.to(device)
        n = embeddings.size(0)
        dist_matrix = torch.cdist(embeddings, embeddings, p=2)

        loss = 0.0
        count = 0

        for i in range(n):
            label = labels[i]
            pos_mask = (labels == label) & (torch.arange(n, device=device) != i)
            neg_mask = labels != label

            pos_indices = torch.where(pos_mask)[0]
            neg_indices = torch.where(neg_mask)[0]

            if len(pos_indices) == 0 or len(neg_indices) < 2:
                continue

            pos_idx = pos_indices[torch.randint(len(pos_indices), (1,), device=device)]

            if self.use_hard_mining:
                dists_to_neg = dist_matrix[i, neg_indices]
                k = min(self.num_hard_negatives, len(neg_indices))
                topk_indices = torch.topk(dists_to_neg, k, largest=False).indices
                negs1 = neg_indices[topk_indices]
            else:
                permuted = torch.randperm(len(neg_indices), device=device)
                k = min(self.num_hard_negatives, len(neg_indices))
                negs1 = neg_indices[permuted[:k]]

            for neg1 in negs1:
                neg2_candidates = neg_indices[neg_indices != neg1]
                if len(neg2_candidates) == 0:
                    continue
                dists_n2 = dist_matrix[neg1, neg2_candidates]
                neg2 = neg2_candidates[torch.argmin(dists_n2)]

                d_ap = dist_matrix[i, pos_idx]
                d_an = dist_matrix[i, neg1]
                d_n1n2 = dist_matrix[neg1, neg2]

                loss += F.relu(d_ap - d_an + self.margin1)
                loss += F.relu(d_ap - d_n1n2 + self.margin2)
                count += 1

        if count == 0:
            quad_loss = torch.zeros([], requires_grad=True, device=device)
        else:
            quad_loss = loss / count

        if self.use_center_loss:
            center = self.center_loss(embeddings, labels)
            return quad_loss + self.center_loss_weight * center
        else:
            return quad_loss
