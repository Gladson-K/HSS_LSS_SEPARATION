#!/usr/bin/env python3
"""
JEV (Joint Embedding Vision/Audio Variation) based model
Supports joint embedding for heart-lung sound separation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class JEVModel(nn.Module):
    def __init__(self, input_dim=128, embed_dim=256, num_heads=8, num_layers=4, num_classes=2):
        super(JEVModel, self).__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, 
                                                   dim_feedforward=embed_dim*4, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.joint_embedding = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim)
        )
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim//2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embed_dim//2, num_classes)
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        # x: (batch, seq, dim) or (batch, dim, seq)
        if x.dim() == 3 and x.shape[1] != x.shape[2]:
            x = x.transpose(1, 2)
        x = self.input_proj(x)
        x = self.transformer_encoder(x)
        x = self.joint_embedding(x)
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)
