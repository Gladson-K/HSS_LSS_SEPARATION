#!/usr/bin/env python3
"""
Feature Pipeline Baseline - trained on top of audio pipeline baseline approach
Uses wavelet features like existing implementation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeaturePipelineBaseline(nn.Module):
    """
    Baseline for feature-based input (wavelet/MFCC features)
    """
    def __init__(self, input_channels=128, seq_len=400, num_classes=2):
        super(FeaturePipelineBaseline, self).__init__()
        # Dilated conv blocks similar to original
        dilation_rates = [2**i for i in range(6)] * 2
        layers = []
        in_channels = input_channels
        for dilation_rate in dilation_rates:
            layers.append(nn.Conv1d(in_channels, 32, kernel_size=3, padding='same', dilation=dilation_rate))
            layers.append(nn.BatchNorm1d(32))
            layers.append(nn.ReLU())
            in_channels = 32
        self.dilated_convs = nn.Sequential(*layers)
        self.conv_out = nn.Conv1d(32, 16, kernel_size=3, padding='same')
        self.bn_out = nn.BatchNorm1d(16)
        self.relu_out = nn.ReLU()
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(16, num_classes)
        
    def forward(self, x):
        # x shape: (batch, channels, seq_len)
        x = self.dilated_convs(x)
        x = self.relu_out(self.bn_out(self.conv_out(x)))
        x = self.global_pool(x).squeeze(-1)
        x = self.fc(x)
        return x
