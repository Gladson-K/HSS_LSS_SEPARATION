# Real Training & Fine-tuning

All scripts here train on the **original** `Data Sets` folders with genuine
train/val/test splits and report real metrics (no simulated numbers).

## Dataset discovery

Set `DATASETS_DIR` to the original `Data Sets` directory (defaults to the
sibling `Heart_Lung_Sound_Separation_SIG/Data Sets`).

Tasks:
- **heart**: `N_New` (Normal), `MVP_New` (Mitral Valve Prolapse), `AS_New` (Aortic Stenosis)
- **lung**: `Audio Files` parsed from `<id>_<Diagnosis>,...wav`

## 1. Classical + advanced model comparison (CPU-friendly)

```bash
# quick smoke run
python training/real_train.py --task heart --epochs 3 --max-per-class 20

# fuller run
python training/real_train.py --task both --epochs 20 --max-per-class 120
```

Compares: `NAS-feature`, `JEV`, `GNN`, `Audio-Transformer`, `Raw-audio-CNN`.
Outputs JSON + plots to `experiments/real_results/`:
`heart_accuracy.png`, `heart_loss.png`, `heart_test_accuracy.png`,
`heart_best_confusion.png` (and `lung_*`).

## 2. Fine-tune open-source foundation models (optional, GPU recommended)

```bash
pip install -r requirements-ml.txt
python training/finetune_foundation.py --task heart --backbone ast  --epochs 3
python training/finetune_foundation.py --task lung  --backbone ast  --epochs 3
```

Backbones (all permissive licenses):
- `ast` — BSD-3-Clause
- `wav2vec2` — Apache-2.0
- `panns` — MIT code / CC-BY-4.0 weights

If transformers/torch are absent the script exits cleanly and the classical
pipeline in the product continues to work.
