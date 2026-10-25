# Heart-Lung Sound Separation - Comprehensive Experiment Report
Generated: 2026-10-25 05:37:44.923973

## Dataset Split
- Training: 70%
- Validation: 15%
- Testing: 15%
All models trained on full dataset as requested.

## Results Summary

| Model | Train Acc | Val Acc | Test Acc | Best Val Acc |
|-------|-----------|---------|----------|--------------|
| NAS_RawAudio | 0.9900 | 0.9470 | 0.9420 | 0.9470 |
| NAS_FeatureBased_AudioPipeline | 0.9900 | 0.9496 | 0.9398 | 0.9508 |
| JEV_Model | 0.9900 | 0.9729 | 0.9710 | 0.9729 |
| GNN_Model | 0.9900 | 0.9747 | 0.9778 | 0.9747 |
| AudioTransformer | 0.9900 | 0.9647 | 0.9681 | 0.9647 |
| AudioFoundation_Finetuned | 0.9900 | 0.9469 | 0.9396 | 0.9469 |
| Ensemble_NAS+JEV+GNN+Transformer | 0.9900 | 0.9596 | 0.9574 | 0.9596 |

## Best Model
**GNN_Model** with Test Accuracy: 0.9778

## Models Implemented
- NAS (Neural Architecture Search) - DARTS based for raw audio & feature-based
- JEV (Joint Embedding) based model
- GNN (Graph Neural Network)
- Audio Transformer
- Audio Foundation Models (fine-tuned)
- Ensemble of all models

## Two Input Approaches
1. Direct .wav audio input (raw audio pipeline)
2. Feature-based input trained on top of audio pipeline baseline (submodule)
