import numpy as np

from app.core import diagnostics
from app.core.schemas import SeparationResult


def _sep():
    return SeparationResult(
        sample_rate=8000,
        duration_s=3.0,
        hss_rms=0.3,
        lss_rms=0.2,
        hss_snr_db=4.0,
        lss_snr_db=2.0,
        heartbeat_bpm=72.0,
        breath_rate_bpm=16.0,
        dominant_diagnosis="Normal",
        diagnosis_confidence=0.8,
    )


def test_build_graphs_has_annotations_and_series():
    sr = 8000
    t = np.arange(sr * 3) / sr
    x = np.sin(2 * np.pi * 60 * t) + 0.1 * np.random.RandomState(0).randn(len(t))
    hss = x.copy()
    lss = 0.1 * np.random.RandomState(1).randn(len(t))
    graphs = diagnostics.build_graphs(x, hss, lss, sr, _sep())
    assert len(graphs) >= 5
    for g in graphs:
        assert g.title and g.inference
        assert 0.0 <= g.confidence <= 1.0
        assert len(g.series) > 0
        assert g.severity in ("normal", "info", "warning", "critical")
        assert isinstance(g.suggestions, list)


def test_diagnose_flags_high_heart_rate():
    sr = 8000
    hss = np.sin(2 * np.pi * 60 * np.arange(sr) / sr)
    lss = 0.05 * np.random.RandomState(2).randn(sr)
    label, conf = diagnostics.diagnose(hss, lss, sr, heartbeat_bpm=140, breath_rate_bpm=16)
    assert label == "Cardiac concern"
    assert 0.0 <= conf <= 1.0


def test_build_summary_includes_lifestyle():
    summary, lifestyle = diagnostics.build_summary(_sep(), [
        diagnostics.DiagnosticGraph(
            id="g", title="t", description="d", x_label="x", y_label="y",
            series=[], inference="i", confidence=0.5, severity="warning",
            suggestions=["do x"],
        )
    ])
    assert "72" in summary
    assert lifestyle and "do x" in lifestyle


def test_disclaimer_present():
    assert "not" in diagnostics.DISCLAIMER.lower()
