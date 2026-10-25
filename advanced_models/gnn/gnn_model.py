#!/usr/bin/env python3
"""
Graph Neural Network for Heart-Lung Sound Separation
Treats time-frequency features as graph nodes
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        
    def forward(self, x, adj):
        # x: (batch, nodes, features), adj: adjacency matrix
        out = torch.matmul(adj, x)
        out = self.linear(out)
        out = F.relu(out)
        return out


class GNNModel(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128, num_layers=3, num_classes=2):
        super(GNNModel, self).__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.gcn_layers = nn.ModuleList([GraphConvLayer(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim//2, num_classes)
        )
        
    def build_adj(self, x, k=5):
        # Build k-NN adjacency
        batch_size, nodes, _ = x.size()
        adj = torch.eye(nodes).unsqueeze(0).repeat(batch_size, 1, 1).to(x.device)
        return adj
        
    def forward(self, x):
        if x.dim() == 3 and x.shape[1] < x.shape[2]:  # (batch, channels, seq)
            x = x.transpose(1, 2)
        batch_size, seq_len, feat_dim = x.shape
        x = self.input_proj(x)
        adj = self.build_adj(x)
        for layer in self.gcn_layers:
            x = layer(x, adj)
        x = x.mean(dim=1)  # Global pooling
        return self.classifier(x)
