"""Real dataset loader + feature extraction for Heart/Lung sound experiments.

Works directly on the original ``Data Sets`` folders:
  * Heart task  : N_New (Normal) / MVP_New (Mitral Valve Prolapse) / AS_New (Aortic Stenosis)
  * Lung  task  : ``Audio Files`` parsed from ``<id>_<Diagnosis>,...wav`` filenames

Everything is dependency-light (numpy + scipy). No simulated data.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

SR = 8000
SECONDS = 3.0
SEQ_LEN = 128          # mel time frames
N_MELS = 128


def resolve_data_root() -> Path:
    env = os.getenv("DATASETS_DIR")
    candidates = [
        Path(env) if env else None,
        Path(__file__).resolve().parents[2] / "Heart_Lung_Sound_Separation_SIG" / "Data Sets",
        Path.home() / "Projects" / "Heart_Lung_Sound_Separation_SIG" / "Data Sets",
    ]
    for c in candidates:
        if c and c.exists():
            return c
    raise FileNotFoundError("Data Sets folder not found; set DATASETS_DIR")


@dataclass
class Item:
    path: str
    label: int
    label_name: str


def _norm_label(raw: str) -> str:
    r = raw.strip().lower()
    if r in ("n", "normal", ""):
        return "Normal"
    if "asthma" in r and "fibrosis" in r:
        return "Asthma+Fibrosis"
    if "asthma" in r:
        return "Asthma"
    if "copd" in r:
        return "COPD"
    if "fibrosis" in r:
        return "Lung Fibrosis"
    if "heart" in r:
        return "Heart Failure"
    if "bron" in r:
        return "Bronchitis"
    return raw.strip() or "Other"


def build_heart_index() -> tuple[list[Item], list[str]]:
    root = resolve_data_root()
    mapping = {"N_New": "Normal", "MVP_New": "MVP", "AS_New": "AS"}
    names = ["Normal", "MVP", "AS"]
    items: list[Item] = []
    for folder, name in mapping.items():
        for wav in sorted((root / folder).glob("*.wav")):
            items.append(Item(str(wav), names.index(name), name))
    return items, names


def build_lung_index(max_classes: int = 5) -> tuple[list[Item], list[str]]:
    root = resolve_data_root()
    folder = root / "Audio Files"
    raw: list[tuple[str, str]] = []
    for wav in sorted(folder.glob("*.wav")):
        stem = wav.stem
        head = stem.split(",", 1)[0]
        diag = head.split("_", 1)[1] if "_" in head else head
        raw.append((str(wav), _norm_label(diag)))
    # keep the most common classes
    from collections import Counter

    counts = Counter(lbl for _, lbl in raw).most_common(max_classes)
    keep = [lbl for lbl, _ in counts]
    names = sorted(keep)
    idx = {n: i for i, n in enumerate(names)}
    items = [Item(p, idx[l], l) for p, l in raw if l in idx]
    return items, names


def load_waveform(path: str, sr: int = SR, seconds: float = SECONDS) -> np.ndarray:
    fs, data = wavfile.read(path)
    data = np.asarray(data)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float64)
    if fs != sr:
        g = np.gcd(int(fs), int(sr))
        data = resample_poly(data, sr // g, fs // g)
    n = int(sr * seconds)
    if len(data) >= n:
        data = data[:n]
    else:
        data = np.pad(data, (0, n - len(data)))
    peak = np.max(np.abs(data)) or 1e-9
    return data / peak


def log_mel(wav: np.ndarray, sr: int = SR, n_mels: int = N_MELS,
            frames: int = SEQ_LEN) -> np.ndarray:
    """Simple log-mel spectrogram (n_mels x frames), normalised to [-1, 1]."""
    win = 512
    hop = max(1, (len(wav) - win) // max(1, frames - 1)) if len(wav) > win else win
    window = np.hanning(win)
    s = np.stack([
        np.abs(np.fft.rfft(wav[i:i + win] * window, n=win))
        for i in range(0, max(1, len(wav) - win), hop)
    ])[:frames]
    if s.shape[0] < frames:
        s = np.pad(s, ((0, frames - s.shape[0]), (0, 0)))
    power = s ** 2
    # triangular mel filterbank
    fb = _mel_filterbank(n_mels, win // 2 + 1, sr)
    mel = power @ fb.T
    logmel = np.log10(mel + 1e-9)
    logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-9)
    return logmel.T.astype(np.float32)  # (n_mels, frames)


def _mel_filterbank(n_mels: int, n_freqs: int, sr: int) -> np.ndarray:
    def hz2mel(hz):
        return 2595 * np.log10(1 + hz / 700.0)

    def mel2hz(m):
        return 700 * (10 ** (m / 2595.0) - 1)

    mels = np.linspace(hz2mel(0), hz2mel(sr / 2), n_mels + 2)
    hz = mel2hz(mels)
    bins = np.floor((n_freqs - 1) * hz / (sr / 2)).astype(int)
    fb = np.zeros((n_mels, n_freqs))
    for m in range(1, n_mels + 1):
        l, c, r = bins[m - 1], bins[m], bins[m + 1]
        if c == l:
            c += 1
        if r == c:
            r += 1
        for k in range(l, min(c, n_freqs)):
            fb[m - 1, k] = (k - l) / max(1, c - l)
        for k in range(c, min(r, n_freqs)):
            fb[m - 1, k] = (r - k) / max(1, r - c)
    return fb


def stratified_split(items: list[Item], val: float = 0.15, test: float = 0.15,
                     seed: int = 42) -> tuple[list[Item], list[Item], list[Item]]:
    rng = random.Random(seed)
    by_label: dict[int, list[Item]] = {}
    for it in items:
        by_label.setdefault(it.label, []).append(it)
    train, vali, testi = [], [], []
    for _, group in by_label.items():
        rng.shuffle(group)
        n = len(group)
        n_test = int(n * test)
        n_val = int(n * val)
        testi.extend(group[:n_test])
        vali.extend(group[n_test:n_test + n_val])
        train.extend(group[n_test + n_val:])
    rng.shuffle(train)
    return train, vali, testi


def limit_per_class(items: list[Item], max_per_class: int, seed: int = 0) -> list[Item]:
    rng = random.Random(seed)
    by_label: dict[int, list[Item]] = {}
    for it in items:
        by_label.setdefault(it.label, []).append(it)
    out: list[Item] = []
    for _, group in by_label.items():
        rng.shuffle(group)
        out.extend(group[:max_per_class])
    return out
