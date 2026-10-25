#!/usr/bin/env python3
"""
Audio Pipeline Baseline - for training on top of it (submodule style)
Implements baseline audio processing pipeline for heart-lung sound separation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioPipelineBaseline(nn.Module):
    """
    Baseline audio pipeline for direct .wav input processing
    """
    def __init__(self, input_channels=1, num_classes=2):
        super(AudioPipelineBaseline, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=64, stride=2, padding=31)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=32, stride=2, padding=15)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=16, stride=2, padding=7)
        self.bn3 = nn.BatchNorm1d(256)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=4, stride=4)
        
        self.conv4 = nn.Conv1d(256, 512, kernel_size=8, stride=2, padding=3)
        self.bn4 = nn.BatchNorm1d(512)
        self.relu4 = nn.ReLU()
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.relu4(self.bn4(self.conv4(x)))
        x = self.global_pool(x).squeeze(-1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
