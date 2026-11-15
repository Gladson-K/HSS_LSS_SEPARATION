"""Real (non-simulated) training harness across all model families.

Trains, validates and tests on the original dataset with genuine metrics, then
writes results + plots to ``experiments/real_results``.

Usage:
    python training/real_train.py --task heart --epochs 6 --max-per-class 30
    python training/real_train.py --task lung  --epochs 6 --max-per-class 30

Models compared per task:
    NAS-feature conv, JEV, GNN, Audio-Transformer, Raw-audio CNN
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from real_dataset import (  # noqa: E402
    SECONDS,
    SR,
    build_heart_index,
    build_lung_index,
    limit_per_class,
    load_waveform,
    log_mel,
    stratified_split,
)

from advanced_models.gnn.gnn_model import GNNModel  # noqa: E402
from advanced_models.jev.jev_model import JEVModel  # noqa: E402
from advanced_models.transformer.transformer_model import AudioTransformer  # noqa: E402
from baselines.audio_pipeline_baseline import AudioPipelineBaseline  # noqa: E402
from baselines.feature_pipeline_baseline import FeaturePipelineBaseline  # noqa: E402

OUT = ROOT / "experiments" / "real_results"


class MelDataset(Dataset):
    def __init__(self, items, num_classes, raw=False):
        self.items = items
        self.num_classes = num_classes
        self.raw = raw
        self.cache = [self._prepare(it) for it in items]

    def _prepare(self, it):
        wav = load_waveform(it.path)
        if self.raw:
            return torch.tensor(wav, dtype=torch.float32).unsqueeze(0)
        return torch.tensor(log_mel(wav), dtype=torch.float32)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.cache[i], self.items[i].label


def build_models(num_classes: int, raw_len: int) -> dict[str, nn.Module]:
    return {
        "NAS-feature": FeaturePipelineBaseline(input_channels=128, seq_len=128, num_classes=num_classes),
        "JEV": JEVModel(input_dim=128, embed_dim=128, num_heads=4, num_layers=2, num_classes=num_classes),
        "GNN": GNNModel(input_dim=128, hidden_dim=128, num_layers=2, num_classes=num_classes),
        "Audio-Transformer": AudioTransformer(input_dim=128, d_model=128, nhead=4,
                                              num_encoder_layers=2, num_classes=num_classes),
        "Raw-audio-CNN": AudioPipelineBaseline(num_classes=num_classes),
    }


def run_epoch(model, loader, criterion, opt=None):
    train = opt is not None
    model.train(train)
    total, correct, loss_sum = 0, 0, 0.0
    preds, truths = [], []
    for xb, yb in loader:
        if train:
            opt.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        if train:
            loss.backward()
            opt.step()
        loss_sum += loss.item() * len(yb)
        p = out.argmax(1)
        correct += (p == yb).sum().item()
        total += len(yb)
        preds.extend(p.tolist())
        truths.extend(yb.tolist())
    return loss_sum / max(1, total), correct / max(1, total), truths, preds


def train_task(task: str, epochs: int, max_per_class: int, raw_len: int):
    if task == "heart":
        items, names = build_heart_index()
    else:
        items, names = build_lung_index(max_classes=5)
    items = limit_per_class(items, max_per_class, seed=1)
    tr, va, te = stratified_split(items, seed=42)
    K = len(names)
    print(f"[{task}] classes={names} train={len(tr)} val={len(va)} test={len(te)}")

    feat_tr = MelDataset(tr, K, raw=False)
    feat_va = MelDataset(va, K, raw=False)
    feat_te = MelDataset(te, K, raw=False)
    raw_tr = MelDataset(tr, K, raw=True)
    raw_va = MelDataset(va, K, raw=True)
    raw_te = MelDataset(te, K, raw=True)

    loaders = {
        False: (DataLoader(feat_tr, batch_size=16, shuffle=True),
                DataLoader(feat_va, batch_size=16), DataLoader(feat_te, batch_size=16)),
        True: (DataLoader(raw_tr, batch_size=16, shuffle=True),
               DataLoader(raw_va, batch_size=16), DataLoader(raw_te, batch_size=16)),
    }

    models = build_models(K, raw_len)
    results = {}
    for name, model in models.items():
        is_raw = name == "Raw-audio-CNN"
        tr_ld, va_ld, te_ld = loaders[is_raw]
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        crit = nn.CrossEntropyLoss()
        hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        best_val, best_state = -1.0, None
        t0 = time.time()
        for ep in range(1, epochs + 1):
            trl, tra, _, _ = run_epoch(model, tr_ld, crit, opt)
            val_, vaa, _, _ = run_epoch(model, va_ld, crit)
            hist["train_loss"].append(trl)
            hist["val_loss"].append(val_)
            hist["train_acc"].append(tra)
            hist["val_acc"].append(vaa)
            if vaa > best_val:
                best_val = vaa
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if best_state is not None:
            model.load_state_dict(best_state)
        _, tea, truths, preds = run_epoch(model, te_ld, crit)
        metrics = {
            "test_accuracy": float(accuracy_score(truths, preds)),
            "precision": float(precision_score(truths, preds, average="macro", zero_division=0)),
            "recall": float(recall_score(truths, preds, average="macro", zero_division=0)),
            "f1": float(f1_score(truths, preds, average="macro", zero_division=0)),
            "best_val_acc": float(best_val),
            "train_seconds": round(time.time() - t0, 1),
            "params": int(sum(p.numel() for p in model.parameters())),
            "history": hist,
            "confusion": confusion_matrix(truths, preds).tolist(),
        }
        results[name] = metrics
        print(f"  {name:18s} test_acc={metrics['test_accuracy']:.3f} "
              f"f1={metrics['f1']:.3f} val={metrics['best_val_acc']:.3f} "
              f"({metrics['train_seconds']}s)")

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{task}_results.json", "w") as f:
        json.dump({"task": task, "classes": names,
                   "split": {"train": len(tr), "val": len(va), "test": len(te)},
                   "results": results}, f, indent=2)
    _plots(task, names, results)
    return results


def _plots(task, names, results):
    # accuracy curves
    plt.figure(figsize=(11, 6))
    for name, m in results.items():
        plt.plot(m["history"]["train_acc"], "--", alpha=0.6, label=f"{name} train")
        plt.plot(m["history"]["val_acc"], "-", linewidth=2, label=f"{name} val")
    plt.title(f"{task.title()} task - accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / f"{task}_accuracy.png", dpi=140)
    plt.close()

    # loss curves
    plt.figure(figsize=(11, 6))
    for name, m in results.items():
        plt.plot(m["history"]["train_loss"], "--", alpha=0.6, label=f"{name} train")
        plt.plot(m["history"]["val_loss"], "-", linewidth=2, label=f"{name} val")
    plt.title(f"{task.title()} task - loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / f"{task}_loss.png", dpi=140)
    plt.close()

    # test accuracy bar
    plt.figure(figsize=(9, 5))
    names_l = list(results)
    accs = [results[n]["test_accuracy"] for n in names_l]
    bars = plt.bar(range(len(names_l)), accs, color="#3385fc")
    plt.xticks(range(len(names_l)), names_l, rotation=25, ha="right", fontsize=9)
    plt.ylim(0, 1.05)
    plt.title(f"{task.title()} task - test accuracy")
    for b, a in zip(bars, accs):
        plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{a:.2f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / f"{task}_test_accuracy.png", dpi=140)
    plt.close()

    # confusion matrix for best model
    best = max(results, key=lambda n: results[n]["test_accuracy"])
    cm = np.array(results[best]["confusion"])
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"{task.title()} - {best} confusion")
    plt.xlabel("predicted")
    plt.ylabel("true")
    ticks = range(len(names))
    plt.xticks(ticks, names, rotation=30, ha="right", fontsize=8)
    plt.yticks(ticks, names, fontsize=8)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / f"{task}_best_confusion.png", dpi=140)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["heart", "lung", "both"], default="both")
    ap.add_argument("--epochs", type=int, default=int(os.getenv("EPOCHS", 6)))
    ap.add_argument("--max-per-class", type=int, default=int(os.getenv("MAX_PER_CLASS", 30)))
    args = ap.parse_args()

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    raw_len = int(SR * SECONDS)
    summary = {}
    tasks = ["heart", "lung"] if args.task == "both" else [args.task]
    for task in tasks:
        summary[task] = train_task(task, args.epochs, args.max_per_class, raw_len)
    print(json.dumps({t: {m: round(v["test_accuracy"], 3) for m, v in r.items()}
                      for t, r in summary.items()}, indent=2))


if __name__ == "__main__":
    main()
