#!/usr/bin/env python3
"""
Ensemble of all models for best performance
Combines NAS models, JEV, GNN, Transformers, Foundation models
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EnsembleModel(nn.Module):
    def __init__(self, models, weights=None):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)
        num_models = len(models)
        if weights is None:
            self.weights = nn.Parameter(torch.ones(num_models) / num_models)
        else:
            self.weights = nn.Parameter(torch.tensor(weights))
        
    def forward(self, x):
        outputs = []
        for model in self.models:
            try:
                outputs.append(model(x))
            except:
                pass
        if not outputs:
            return None
        weights = F.softmax(self.weights[:len(outputs)], dim=0)
        ensemble_out = sum(w * out for w, out in zip(weights, outputs))
        return ensemble_out
