"""Shared pytest fixtures."""
from __future__ import annotations

import io

import numpy as np
import pytest
from scipy.io import wavfile


def synth_audio(sr: int = 8000, seconds: float = 3.0, heart_hz: float = 1.2,
                carrier_hz: float = 70.0, noise: float = 0.05) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    pulses = (np.sin(2 * np.pi * heart_hz * t) > 0.5).astype(float)
    heart = 0.6 * np.sin(2 * np.pi * carrier_hz * t) * pulses
    breath = 0.2 * np.sin(2 * np.pi * (carrier_hz + 300) * t) * np.sin(2 * np.pi * 0.25 * t)
    x = heart + breath + noise * np.random.RandomState(0).randn(len(t))
    x = x / (np.max(np.abs(x)) + 1e-9)
    return (x * 30000).astype(np.int16)


@pytest.fixture
def wav_bytes() -> bytes:
    sr = 8000
    data = synth_audio(sr=sr)
    buf = io.BytesIO()
    wavfile.write(buf, sr, data)
    return buf.getvalue()
