#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  1 19:19:55 2025

@author: mdb
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class QuadrupletLoss(nn.Module):
    def __init__(self, margin1=0.3, margin2=0.3, max_quadruplets=10):
        super().__init__()
        self.margin1 = margin1
        self.margin2 = margin2
        self.max_quadruplets = max_quadruplets

    def forward(self, embeddings, labels):
        device = embeddings.device
        labels = labels.to(device)
        n = embeddings.size(0)

        # Distancias euclídeas entre todos los embeddings
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

            # Selecciona un positivo aleatorio
            pos_idx = pos_indices[torch.randint(len(pos_indices), (1,), device=device)]

            # Selecciona pares negativos diferentes
            neg_indices = neg_indices[torch.randperm(len(neg_indices), device=device)]
            neg_pairs = []
            for a in range(len(neg_indices)):
                for b in range(a + 1, len(neg_indices)):
                    neg_pairs.append((neg_indices[a], neg_indices[b]))
                    if len(neg_pairs) >= self.max_quadruplets:
                        break
                if len(neg_pairs) >= self.max_quadruplets:
                    break

            # Distancia ancla-positivo
            d_ap = dist_matrix[i, pos_idx]

            for neg1, neg2 in neg_pairs:
                d_an = dist_matrix[i, neg1]
                d_n1n2 = dist_matrix[neg1, neg2]
                loss += F.relu(d_ap - d_an + self.margin1)
                loss += F.relu(d_ap - d_n1n2 + self.margin2)
                count += 1

        if count == 0:
            return torch.zeros([], requires_grad=True, device=device)

        return loss / count
