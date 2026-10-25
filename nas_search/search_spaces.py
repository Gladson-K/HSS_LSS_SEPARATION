#!/usr/bin/env python3
"""
Search spaces for NAS - two approaches:
1. Raw audio waveform input approach
2. Feature-based input approach (wavelet features)
"""
import torch
import torch.nn as nn


class RawAudioSearchSpace(nn.Module):
    """Search space for direct .wav audio input"""
    def __init__(self, num_classes=2):
        super(RawAudioSearchSpace, self).__init__()
        # Flexible architecture
        self.stem = nn.Sequential(
            nn.Conv1d(1, 48, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(48),
            nn.ReLU()
        )
        
    def forward(self, x):
        x = self.stem(x)
        return x


class FeatureBasedSearchSpace(nn.Module):
    """Search space for feature-based input (wavelet features)"""
    def __init__(self, in_channels=128, num_classes=2):
        super(FeatureBasedSearchSpace, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
    def forward(self, x):
        x = self.stem(x)
        return x
