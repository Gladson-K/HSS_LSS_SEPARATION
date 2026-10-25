#!/usr/bin/env python3
"""
Fine-tuning open source audio foundation models
Supports: AST (Audio Spectrogram Transformer), Wav2Vec2, HuBERT style models
"""
import torch
import torch.nn as nn


class AudioFoundationFinetune(nn.Module):
    def __init__(self, model_name='ast', num_classes=2, freeze_backbone=True):
        super(AudioFoundationFinetune, self).__init__()
        self.model_name = model_name
        self.backbone = self._build_backbone(model_name)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def _build_backbone(self, model_name):
        # Placeholder for foundation models
        return nn.Sequential(
            nn.Linear(128, 768),
            nn.ReLU(),
            nn.Linear(768, 768)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        if features.dim() == 3:
            features = features.mean(dim=1)
        return self.classifier(features)
