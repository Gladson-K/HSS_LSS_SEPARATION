"""Fine-tune open-source audio foundation models (AST / PANNs / wav2vec2) on the
heart/lung tasks.

Requires optional heavy deps (install with ``pip install -r requirements-ml.txt``):
    pip install torch torchaudio transformers audiotimm

If the deps or weights are unavailable the script prints a clear message and
exits cleanly (code 0) so CI / the product keep working without a GPU.

The selected models are all permissive-licensed (see app.core.model_registry):
    AST          -> BSD-3-Clause
    PANNs CNN14  -> MIT code / CC-BY-4.0 weights
    wav2vec2     -> Apache-2.0
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))

from real_dataset import (  # noqa: E402
    SR,
    build_heart_index,
    build_lung_index,
    limit_per_class,
    load_waveform,
    stratified_split,
)

SUPPORTED = {
    "ast": ("MIT/ast-finetuned-audioset-10-10-0.4593", "BSD-3-Clause"),
    "wav2vec2": ("facebook/wav2vec2-base", "Apache-2.0"),
    "panns": ("qiuqiangkong/audioset_tagging_cnn", "MIT/CC-BY-4.0"),
}


def _have(pkg: str) -> bool:
    return importlib.util.find_spec(pkg) is not None


def _require_deps() -> bool:
    missing = [p for p in ("torch", "transformers") if not _have(p)]
    if missing:
        print(
            "Missing optional dependencies: " + ", ".join(missing) + "\n"
            "Install them with:  pip install -r requirements-ml.txt\n"
            "Skipping foundation-model fine-tuning (classical pipeline remains active)."
        )
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["heart", "lung"], default="heart")
    ap.add_argument("--backbone", choices=list(SUPPORTED), default="ast")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-per-class", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", default=str(ROOT / "models"))
    args = ap.parse_args()

    hf_name, license_ = SUPPORTED[args.backbone]
    print(f"Backbone: {args.backbone} ({hf_name})  License: {license_}")

    if not _require_deps():
        return 0

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

    if args.task == "heart":
        items, names = build_heart_index()
    else:
        items, names = build_lung_index(max_classes=5)
    items = limit_per_class(items, args.max_per_class, seed=1)
    tr, va, te = stratified_split(items, seed=42)
    K = len(names)
    print(f"[{args.task}] classes={names} train={len(tr)} val={len(va)} test={len(te)}")

    if args.backbone not in ("ast", "wav2vec2"):
        print(f"Note: '{args.backbone}' requires the audiotimm loader; falling back to AST.")
        hf_name = "MIT/ast-finetuned-audioset-10-10-0.4593"

    cache = os.getenv("MODEL_CACHE_DIR", "/tmp/cardiopulmo_models")
    fe = AutoFeatureExtractor.from_pretrained(hf_name, cache_dir=cache)
    target_sr = fe.sampling_rate

    class Ds(Dataset):
        def __init__(self, its):
            self.its = its
            self.data = [load_waveform(it.path, sr=target_sr, seconds=3.0) for it in its]

        def __len__(self):
            return len(self.its)

        def __getitem__(self, i):
            return torch.tensor(self.data[i], dtype=torch.float32), self.its[i].label

    model = AutoModelForAudioClassification.from_pretrained(
        hf_name, num_labels=K, ignore_mismatched_sizes=True, cache_dir=cache
    )
    model.config.id2label = {i: n for i, n in enumerate(names)}
    model.config.label2id = {n: i for i, n in enumerate(names)}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    ld_tr = DataLoader(Ds(tr), batch_size=args.batch_size, shuffle=True)
    ld_va = DataLoader(Ds(va), batch_size=args.batch_size)
    ld_te = DataLoader(Ds(te), batch_size=args.batch_size)

    def collate(batch):
        wavs, labels = zip(*batch)
        inputs = fe(list(wavs), sampling_rate=target_sr, return_tensors="pt", padding=True)
        return inputs, torch.tensor(labels)

    def run(loader, train=False):
        model.train(train)
        correct = total = 0
        for batch in loader:
            inputs, labels = collate(batch)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)
            if train:
                opt.zero_grad()
            out = model(**inputs).logits
            loss = crit(out, labels)
            if train:
                loss.backward()
                opt.step()
            correct += (out.argmax(1) == labels).sum().item()
            total += len(labels)
        return correct / max(1, total)

    best, best_state = 0.0, None
    for ep in range(1, args.epochs + 1):
        va_acc = 0.0
        tr_acc = run(ld_tr, train=True)
        va_acc = run(ld_va)
        print(f"epoch {ep}/{args.epochs}  train_acc={tr_acc:.3f}  val_acc={va_acc:.3f}")
        if va_acc >= best:
            best = va_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    te_acc = run(ld_te)
    print(f"test_acc={te_acc:.3f}  (best val={best:.3f})")

    os.makedirs(args.out, exist_ok=True)
    dest = os.path.join(args.out, f"finetuned_{args.backbone}_{args.task}")
    model.save_pretrained(dest)
    fe.save_pretrained(dest)
    with open(os.path.join(dest, "LICENSE.txt"), "w") as f:
        f.write(f"Base model: {hf_name}\nLicense: {license_}\nFine-tuned by CardioPulmo AI.\n")
    print(f"Saved fine-tuned model -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
