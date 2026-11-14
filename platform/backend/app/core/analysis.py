"""High-level analysis orchestration + in-memory result store."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from app.config import get_settings
from app.core import audio_pipeline as ap
from app.core import diagnostics
from app.core.schemas import AnalysisResponse, SeparationResult

# Bounded in-memory store of recent analyses (keyed by analysis_id).
_STORE: dict[str, AnalysisResponse] = {}
_STORE_ORDER: list[str] = []
_MAX_STORE = 64


def _remember(resp: AnalysisResponse) -> None:
    _STORE[resp.analysis_id] = resp
    _STORE_ORDER.append(resp.analysis_id)
    while len(_STORE_ORDER) > _MAX_STORE:
        old = _STORE_ORDER.pop(0)
        _STORE.pop(old, None)


def get_analysis(analysis_id: str) -> AnalysisResponse | None:
    return _STORE.get(analysis_id)


def run_analysis(raw: bytes | str, filename: str = "recording.wav") -> AnalysisResponse:
    settings = get_settings()
    audio = ap.load_audio(raw, target_sr=settings.sample_rate)
    x, sr = audio.samples, audio.sample_rate

    hss, lss = ap.separate(x, sr, hss_band=settings.hss_band, lss_cutoff=settings.lss_cutoff)

    hss_rms = float((hss ** 2).mean() ** 0.5)
    lss_rms = float((lss ** 2).mean() ** 0.5)

    hss_snr = ap.snr_db(hss, x - hss)
    lss_snr = ap.snr_db(lss, x - lss)

    heartbeat_bpm = ap.estimate_rate_bpm(hss, sr, min_bpm=40, max_bpm=200)
    breath_rate_bpm = ap.estimate_rate_bpm(lss, sr, min_bpm=6, max_bpm=60)

    label, conf = diagnostics.diagnose(hss, lss, sr, heartbeat_bpm, breath_rate_bpm)

    sep = SeparationResult(
        sample_rate=sr,
        duration_s=round(audio.duration_s, 3),
        hss_rms=round(hss_rms, 6),
        lss_rms=round(lss_rms, 6),
        hss_snr_db=round(float(hss_snr), 2),
        lss_snr_db=round(float(lss_snr), 2),
        heartbeat_bpm=round(heartbeat_bpm, 1) if heartbeat_bpm else None,
        breath_rate_bpm=round(breath_rate_bpm, 1) if breath_rate_bpm else None,
        dominant_diagnosis=label,
        diagnosis_confidence=round(conf, 3),
    )

    graphs = diagnostics.build_graphs(x, hss, lss, sr, sep)
    summary, lifestyle = diagnostics.build_summary(sep, graphs)

    resp = AnalysisResponse(
        analysis_id=uuid.uuid4().hex,
        filename=filename,
        created_at=datetime.now(timezone.utc).isoformat(),
        separation=sep,
        graphs=graphs,
        summary=summary,
        lifestyle=lifestyle,
        disclaimer=diagnostics.DISCLAIMER,
    )
    _remember(resp)
    return resp
