"""Signal-processing core for heart / lung sound analysis.

This module is deliberately dependency-light (numpy + scipy + PyWavelets) so it
runs on a plain CPU and inside the test suite. Expensive deep-learning models are
layered on top in :mod:`app.ml.foundation_models` and are optional.
"""
from __future__ import annotations

import io
import math
from dataclasses import dataclass

import numpy as np
import scipy.signal as sig

try:  # PyWavelets is optional at import time
    import pywt
except Exception:  # pragma: no cover
    pywt = None


@dataclass
class LoadedAudio:
    samples: np.ndarray
    sample_rate: int

    @property
    def duration_s(self) -> float:
        return len(self.samples) / float(self.sample_rate)


def _to_mono(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data.astype(np.float64)


def load_audio(raw: bytes | str, target_sr: int = 8000) -> LoadedAudio:
    """Load audio from a file path or raw bytes and resample to ``target_sr``."""
    if isinstance(raw, (bytes, bytearray)):
        buf = io.BytesIO(bytes(raw))
        try:
            from scipy.io import wavfile

            sr, data = wavfile.read(buf)
        except Exception:
            # Fall back to soundfile for compressed formats if present
            try:
                import soundfile as sf  # type: ignore

                buf.seek(0)
                data, sr = sf.read(buf)
            except Exception as exc:  # pragma: no cover
                raise ValueError("Unsupported or corrupt audio file") from exc
    else:
        from scipy.io import wavfile

        sr, data = wavfile.read(raw)

    x = _to_mono(data)
    if np.issubdtype(np.asarray(data).dtype, np.integer):
        max_val = float(np.iinfo(np.asarray(data).dtype).max)
        x = x / max_val

    if sr != target_sr:
        g = math.gcd(int(sr), int(target_sr))
        x = sig.resample_poly(x, target_sr // g, sr // g)
        sr = target_sr

    x = x - np.mean(x)
    peak = np.max(np.abs(x)) or 1e-9
    x = x / peak
    return LoadedAudio(samples=x, sample_rate=int(sr))


def kaiser_bpf(x: np.ndarray, sr: int, low: float, high: float, numtaps: int = 101,
               beta: float = 8.6) -> np.ndarray:
    nyq = 0.5 * sr
    taps = sig.firwin(numtaps, [low / nyq, high / nyq], pass_zero=False,
                      window=("kaiser", beta))
    return sig.lfilter(taps, 1.0, x)


def kaiser_hpf(x: np.ndarray, sr: int, cutoff: float, numtaps: int = 101,
               beta: float = 8.6) -> np.ndarray:
    nyq = 0.5 * sr
    taps = sig.firwin(numtaps, cutoff / nyq, pass_zero=False,
                      window=("kaiser", beta))
    return sig.lfilter(taps, 1.0, x)


def kaiser_lpf(x: np.ndarray, sr: int, cutoff: float, numtaps: int = 101,
               beta: float = 8.6) -> np.ndarray:
    nyq = 0.5 * sr
    taps = sig.firwin(numtaps, cutoff / nyq, pass_zero=True, window=("kaiser", beta))
    return sig.lfilter(taps, 1.0, x)


def daubechies_band(x: np.ndarray, wavelet: str = "db4", level: int = 4,
                    keep: str = "mid") -> np.ndarray:
    """Wavelet-domain band selection using Daubechies decomposition.

    ``keep`` is one of ``low`` (approximation), ``mid`` (details), ``high``.
    """
    if pywt is None:
        return x
    coeffs = pywt.wavedec(x, wavelet, mode="symmetric", level=level)
    out = [np.zeros_like(c) for c in coeffs]
    if keep == "low":
        out[0] = coeffs[0]
    elif keep == "high":
        if len(coeffs) > 1:
            out[1] = coeffs[1]
    else:  # mid
        for i in range(1, max(2, len(coeffs) - 1)):
            out[i] = coeffs[i]
    rec = pywt.waverec(out, wavelet, mode="symmetric")
    return rec[: len(x)] if len(rec) >= len(x) else np.pad(rec, (0, len(x) - len(rec)))


def rls_filter(x: np.ndarray, order: int = 4, forgetting: float = 0.99) -> np.ndarray:
    """Recursive Least Squares adaptive line enhancer (vectorised inner loop)."""
    n = len(x)
    if n <= order:
        return x.copy()
    w = np.zeros(order)
    P = np.eye(order) / 0.01
    y = np.zeros(n)
    for i in range(order, n):
        xv = x[i - order : i][::-1]
        yv = float(np.dot(w, xv))
        err = x[i] - yv
        Px = P @ xv
        k = Px / (forgetting + float(xv @ Px))
        w = w + k * err
        P = (P - np.outer(k, Px)) / forgetting
        y[i] = yv
    y[:order] = x[:order]
    return y


def band_energy(x: np.ndarray, sr: int, low: float, high: float) -> float:
    if len(x) < 8:
        return 0.0
    f, pxx = sig.welch(x, fs=sr, nperseg=min(1024, len(x)))
    mask = (f >= low) & (f <= high)
    if not mask.any():
        return 0.0
    integrate = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(integrate(pxx[mask], f[mask]))


def snr_db(signal: np.ndarray, noise: np.ndarray) -> float:
    p_sig = float(np.mean(signal ** 2)) or 1e-12
    p_noise = float(np.mean(noise ** 2)) or 1e-12
    return 10.0 * math.log10(p_sig / p_noise)


def estimate_rate_bpm(envelope: np.ndarray, sr: int, min_bpm: float = 30,
                      max_bpm: float = 240) -> float | None:
    """Estimate a periodic rate from an amplitude envelope via autocorrelation."""
    env = np.abs(sig.hilbert(envelope)) if len(envelope) > 16 else np.abs(envelope)
    env = env - np.mean(env)
    if not np.any(env):
        return None
    ac = np.correlate(env, env, mode="full")[len(env) - 1 :]
    min_lag = int(sr * 60.0 / max_bpm)
    max_lag = int(sr * 60.0 / min_bpm)
    if max_lag >= len(ac) or max_lag <= min_lag:
        return None
    window = ac[min_lag:max_lag]
    if not np.any(window):
        return None
    lag = min_lag + int(np.argmax(window))
    if lag <= 0:
        return None
    bpm = 60.0 * sr / lag
    return float(min(max(bpm, min_bpm), max_bpm))


def separate(x: np.ndarray, sr: int, hss_band: tuple[float, float] = (20.0, 200.0),
             lss_cutoff: float = 150.0) -> tuple[np.ndarray, np.ndarray]:
    """Separate heart (HSS) and lung (LSS) components following the research
    algorithm: Kaiser BPF/HPF -> Daubechies band -> RLS."""
    # Heart: band-pass into 20-200 Hz, wavelet mid-band, RLS enhancement
    hss = kaiser_bpf(x, sr, hss_band[0], hss_band[1])
    hss = daubechies_band(hss, "db4", level=4, keep="mid")
    hss = rls_filter(hss)

    # Lung: high-pass above cutoff, wavelet high-band, gentle RLS
    lss = kaiser_hpf(x, sr, lss_cutoff)
    lss = kaiser_lpf(lss, sr, min(sr / 2 - 1, 4200))
    lss = daubechies_band(lss, "db6", level=5, keep="high")
    lss = rls_filter(lss)

    return hss.astype(np.float64), lss.astype(np.float64)
