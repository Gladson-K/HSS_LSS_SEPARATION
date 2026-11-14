"""Optional wrappers around open-source audio foundation models.

These are lazy/optional: if ``transformers``/``torch``/``torchaudio`` are not
installed the functions return ``None`` and the API falls back to the classical
signal-processing pipeline. This keeps the core service lightweight while still
allowing a GPU/full install to use SOTA backbones.

Models are loaded from the Hugging Face hub (permissive licenses flagged in the
model registry) and cached under ``settings.model_cache_dir``.
"""
from __future__ import annotations

import os
from typing import Any

from app.config import get_settings

_CACHE: dict[str, Any] = {}


def _have(*pkgs: str) -> bool:
    import importlib.util

    return all(importlib.util.find_spec(p) is not None for p in pkgs)


def available_backends() -> list[str]:
    backends = []
    if _have("transformers", "torch"):
        backends.append("transformers")
    if _have("audiotimm"):
        backends.append("audiotimm")
    return backends


def load_ast():
    """Load the Audio Spectrogram Transformer (BSD-3-Clause)."""
    if not _have("transformers", "torch"):
        return None
    key = "ast"
    if key in _CACHE:
        return _CACHE[key]
    from transformers import ASTFeatureExtractor, ASTForAudioClassification  # type: ignore

    os.makedirs(get_settings().model_cache_dir, exist_ok=True)
    name = "MIT/ast-finetuned-audioset-10-10-0.4593"
    feat = ASTFeatureExtractor.from_pretrained(name, cache_dir=get_settings().model_cache_dir)
    model = ASTForAudioClassification.from_pretrained(name, cache_dir=get_settings().model_cache_dir)
    model.eval()
    _CACHE[key] = (feat, model)
    return _CACHE[key]


def classify_with_ast(samples, sample_rate: int, top_k: int = 5) -> list[dict] | None:
    """Return top-k AudioSet tags for a numpy waveform, or None if unavailable."""
    bundle = load_ast()
    if bundle is None:
        return None
    import torch  # type: ignore

    feat, model = bundle
    inputs = feat(samples, sampling_rate=sample_rate, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.sigmoid(logits)[0]
    idx = torch.topk(probs, k=top_k).indices.tolist()
    id2label = model.config.id2label
    return [{"label": id2label[i], "score": float(probs[i])} for i in idx]


def embeddings_available() -> bool:
    return _have("transformers", "torch")
