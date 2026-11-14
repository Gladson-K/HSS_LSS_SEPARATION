"""Diagnostic graph generation and interpretation.

Turns a separated heart/lung signal into a list of annotated graphs. Each graph
carries a plain-language inference, a confidence, a severity and lifestyle
suggestions so the GUI can page through them with left/right navigation.
"""
from __future__ import annotations

import numpy as np
import scipy.signal as sig

from app.core import audio_pipeline as ap
from app.core.schemas import DiagnosticGraph, GraphPoint, SeparationResult

MAX_POINTS = 240

DISCLAIMER = (
    "This tool provides research-grade signal analysis for educational and "
    "screening support only. It is NOT a medical device and does not provide a "
    "diagnosis. Always consult a qualified clinician for medical decisions."
)


def _downsample(y: np.ndarray, n: int = MAX_POINTS) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)
    if len(y) > n:
        idx = np.linspace(0, len(y) - 1, n).astype(int)
        y = y[idx]
    x = np.linspace(0.0, 1.0, len(y))
    return x, y


def _series(x: np.ndarray, y: np.ndarray, scale: float = 1.0) -> list[GraphPoint]:
    xs, ys = _downsample(y)
    return [GraphPoint(t=float(a), value=float(b) * scale) for a, b in zip(xs, ys)]


def _envelope(x: np.ndarray, sr: int, win_ms: float = 50.0) -> np.ndarray:
    win = max(1, int(sr * win_ms / 1000.0))
    env = np.convolve(np.abs(x), np.ones(win) / win, mode="same")
    return env


def _spectrum(x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    f, pxx = sig.welch(x, fs=sr, nperseg=min(1024, len(x)))
    return f, pxx


def _normalise(y: np.ndarray) -> np.ndarray:
    m = np.max(np.abs(y)) or 1e-9
    return y / m


def build_graphs(x: np.ndarray, hss: np.ndarray, lss: np.ndarray, sr: int,
                 sep: SeparationResult) -> list[DiagnosticGraph]:
    graphs: list[DiagnosticGraph] = []
    t_end = len(x) / sr

    # 1. Raw auscultation waveform
    graphs.append(
        DiagnosticGraph(
            id="waveform",
            title="Auscultation Waveform",
            description="Normalised raw stethoscope signal after DC removal.",
            x_label="Time (normalised)",
            y_label="Amplitude",
            series=_series(*_downsample(_normalise(x))),
            inference=(
                "Overall signal captured cleanly with stable amplitude."
                if np.std(x) > 0.01
                else "Very low amplitude input - reposition the stethoscope or increase gain."
            ),
            confidence=0.95,
            severity="normal" if np.std(x) > 0.01 else "warning",
            suggestions=[
                "Hold the stethoscope firmly against bare skin to reduce friction noise.",
                "Ask the patient to breathe quietly for the first few seconds.",
            ],
        )
    )

    # 2. Separated heart (HSS) envelope
    hss_env = _envelope(hss, sr)
    graphs.append(
        DiagnosticGraph(
            id="hss_envelope",
            title="Heart Sound (HSS) Envelope",
            description="Amplitude envelope of the separated heart component (S1/S2).",
            x_label="Time (normalised)",
            y_label="Envelope",
            series=_series(*_downsample(_normalise(hss_env))),
            inference=(
                f"Heart rate estimated at {sep.heartbeat_bpm:.0f} BPM."
                if sep.heartbeat_bpm
                else "Heart rhythm could not be estimated reliably from this segment."
            ),
            confidence=0.85 if sep.heartbeat_bpm else 0.4,
            severity=_hr_severity(sep.heartbeat_bpm),
            suggestions=_hr_suggestions(sep.heartbeat_bpm),
        )
    )

    # 3. Separated lung (LSS) envelope
    lss_env = _envelope(lss, sr)
    graphs.append(
        DiagnosticGraph(
            id="lss_envelope",
            title="Lung Sound (LSS) Envelope",
            description="Amplitude envelope of the separated lung component (breath cycles).",
            x_label="Time (normalised)",
            y_label="Envelope",
            series=_series(*_downsample(_normalise(lss_env))),
            inference=(
                f"Respiratory rate estimated at {sep.breath_rate_bpm:.0f} breaths/min."
                if sep.breath_rate_bpm
                else "Respiratory cycles not clearly periodic in this segment."
            ),
            confidence=0.8 if sep.breath_rate_bpm else 0.4,
            severity=_rr_severity(sep.breath_rate_bpm),
            suggestions=_rr_suggestions(sep.breath_rate_bpm),
        )
    )

    # 4. Frequency spectrum with heart/lung bands
    f, pxx = _spectrum(x, sr)
    db = 10 * np.log10(pxx + 1e-12)
    graphs.append(
        DiagnosticGraph(
            id="spectrum",
            title="Power Spectrum (HSS vs LSS bands)",
            description="Spectral energy. Heart sounds dominate 20-200 Hz, lungs above 150 Hz.",
            x_label="Frequency (normalised)",
            y_label="Power (dB)",
            series=_series(*_downsample(db)),
            inference=_spectrum_inference(f, pxx),
            confidence=0.8,
            severity="info",
            suggestions=[
                "Lung energy above 400 Hz may indicate wheeze or crackles - correlate clinically.",
                "Persistent low-frequency peaks may reflect ambient/mechanical noise.",
            ],
        )
    )

    # 5. Heart vs lung energy balance
    hss_e = float(np.mean(hss ** 2)) or 1e-12
    lss_e = float(np.mean(lss ** 2)) or 1e-12
    ratio_db = 10 * np.log10(hss_e / lss_e)
    graphs.append(
        DiagnosticGraph(
            id="energy_balance",
            title="Heart vs Lung Energy Balance",
            description="Relative energy of separated components across the recording.",
            x_label="Segment (normalised)",
            y_label="Relative energy",
            series=_series(*_downsample(_normalise(np.abs(_segment_energy(x, sr))))),
            inference=(
                f"Heart-dominant recording ({ratio_db:+.1f} dB). "
                if ratio_db > 3
                else f"Lung-dominant recording ({ratio_db:+.1f} dB). "
                if ratio_db < -3
                else f"Balanced heart/lung mix ({ratio_db:+.1f} dB). "
            )
            + "Isolation quality: "
            + ("good." if sep.hss_snr_db > 0 and sep.lss_snr_db > 0 else "moderate - consider a longer recording."),
            confidence=0.78,
            severity="normal",
            suggestions=[
                "For cleaner heart isolation, record in a quiet room and hold breath briefly.",
                "For cleaner lung isolation, use a higher low-frequency cutoff.",
            ],
        )
    )

    # 6. Respiration waveform proxy (LSS low-pass)
    breath = ap.kaiser_lpf(lss, sr, 2.0)
    graphs.append(
        DiagnosticGraph(
            id="respiration",
            title="Respiration Trend",
            description="Slow low-pass respiratory trend extracted from the lung component.",
            x_label="Time (normalised)",
            y_label="Trend",
            series=_series(*_downsample(_normalise(breath))),
            inference=(
                "Respiratory trend shows regular cycles."
                if sep.breath_rate_bpm and 10 <= sep.breath_rate_bpm <= 30
                else "Respiratory trend irregular or extrapolated - capture a longer quiet breath sample."
            ),
            confidence=0.7,
            severity=_rr_severity(sep.breath_rate_bpm),
            suggestions=_rr_suggestions(sep.breath_rate_bpm),
        )
    )

    return graphs


def _segment_energy(x: np.ndarray, sr: int, segments: int = 64) -> np.ndarray:
    seg = max(1, len(x) // segments)
    return np.array([
        np.mean(x[i * seg:(i + 1) * seg] ** 2) for i in range(segments)
    ])


def _hr_severity(bpm: float | None) -> str:
    if bpm is None:
        return "info"
    if bpm < 50 or bpm > 120:
        return "critical"
    if bpm < 60 or bpm > 100:
        return "warning"
    return "normal"


def _rr_severity(rr: float | None) -> str:
    if rr is None:
        return "info"
    if rr < 8 or rr > 30:
        return "critical"
    if rr < 12 or rr > 20:
        return "warning"
    return "normal"


def _hr_suggestions(bpm: float | None) -> list[str]:
    if bpm is None:
        return ["Record a longer segment (>=10s) at rest to estimate heart rate."]
    if bpm > 100:
        return [
            "Elevated resting heart rate - rest 5 minutes and re-measure.",
            "Reduce caffeine and maintain regular aerobic exercise.",
            "If persistently high, consult a cardiologist.",
        ]
    if bpm < 60:
        return [
            "Low resting heart rate - normal in athletes, otherwise evaluate with a clinician.",
            "Stay hydrated and avoid sudden posture changes.",
        ]
    return ["Heart rate is within the typical resting range.", "Continue regular activity and balanced sleep."]


def _rr_suggestions(rr: float | None) -> list[str]:
    if rr is None:
        return ["Capture a longer, quiet breathing segment for rate estimation."]
    if rr > 20:
        return [
            "Elevated respiratory rate - rest and re-measure when calm.",
            "Practice slow diaphragmatic breathing (4s in / 6s out).",
            "If breathlessness persists, seek medical advice.",
        ]
    if rr < 12:
        return ["Low respiratory rate - ensure the recording is not over-filtered.", "Re-measure over a full minute."]
    return ["Respiratory rate is within the normal adult range.", "Maintain regular aerobic activity."]


def _spectrum_inference(f: np.ndarray, pxx: np.ndarray) -> str:
    high = float(np.sum(pxx[f > 400])) if np.any(f > 400) else 0.0
    low = float(np.sum(pxx[(f >= 20) & (f <= 200)])) if np.any((f >= 20) & (f <= 200)) else 0.0
    if high > low * 1.5:
        return "High-frequency energy dominates - possible adventitious lung sounds (wheeze/crackles)."
    if low > high * 1.5:
        return "Low-frequency heart energy dominates - consistent with strong S1/S2 components."
    return "Balanced low/high frequency content."


def diagnose(hss: np.ndarray, lss: np.ndarray, sr: int,
             heartbeat_bpm: float | None, breath_rate_bpm: float | None) -> tuple[str, float]:
    """Lightweight rule-based triage label with a confidence score."""
    score: dict[str, float] = {"Normal": 1.0, "Cardiac concern": 0.0, "Respiratory concern": 0.0}

    if heartbeat_bpm is not None:
        if heartbeat_bpm > 100:
            score["Cardiac concern"] += 0.6
        elif heartbeat_bpm < 50:
            score["Cardiac concern"] += 0.4
    if breath_rate_bpm is not None:
        if breath_rate_bpm > 20 or breath_rate_bpm < 10:
            score["Respiratory concern"] += 0.6

    # Adventitious lung-sound energy proxy
    f, pxx = _spectrum(lss, sr)
    adv = float(np.sum(pxx[(f >= 400) & (f <= 2000)]))
    total = float(np.sum(pxx)) or 1e-12
    if adv / total > 0.35:
        score["Respiratory concern"] += 0.5
        score["Normal"] *= 0.5

    label = max(score, key=score.get)
    total_score = sum(score.values()) or 1.0
    confidence = float(min(0.95, max(0.4, score[label] / total_score)))
    return label, confidence


def build_summary(sep: SeparationResult, graphs: list[DiagnosticGraph]) -> tuple[str, list[str]]:
    warnings = [g for g in graphs if g.severity in ("warning", "critical")]
    parts = [
        f"Dominant screen: {sep.dominant_diagnosis} "
        f"(confidence {sep.diagnosis_confidence*100:.0f}%).",
        f"Heart/lung isolation SNR {sep.hss_snr_db:.1f} dB / {sep.lss_snr_db:.1f} dB.",
    ]
    if wrist := (sep.heartbeat_bpm):
        parts.append(f"Estimated heart rate {wrist:.0f} BPM.")
    if rr := (sep.breath_rate_bpm):
        parts.append(f"Estimated respiratory rate {rr:.0f} breaths/min.")
    if warnings:
        parts.append(f"{len(warnings)} graph(s) flagged for attention.")

    lifestyle: list[str] = []
    for g in warnings:
        lifestyle.extend(g.suggestions[:1])
    if not lifestyle:
        lifestyle = [
            "Maintain 150 minutes/week of moderate aerobic exercise.",
            "Eat a heart-healthy, low-sodium diet rich in vegetables and fibre.",
            "Avoid smoking and limit alcohol; ensure good sleep hygiene.",
            "Re-screen annually or as advised by your clinician.",
        ]
    # de-duplicate, keep order
    seen: set[str] = set()
    lifestyle = [s for s in lifestyle if not (s in seen or seen.add(s))]
    return " ".join(parts), lifestyle
