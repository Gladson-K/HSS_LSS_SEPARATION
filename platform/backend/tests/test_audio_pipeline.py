import numpy as np

from app.core import audio_pipeline as ap
from tests.conftest import synth_audio


def test_load_audio_resamples_and_normalises():
    import io
    from scipy.io import wavfile

    data = synth_audio(sr=16000, seconds=2.0)
    buf = io.BytesIO()
    wavfile.write(buf, 16000, data)
    audio = ap.load_audio(buf.getvalue(), target_sr=8000)
    assert audio.sample_rate == 8000
    assert abs(len(audio.samples) - 16000) < 100
    assert np.max(np.abs(audio.samples)) <= 1.0 + 1e-6


def test_separate_returns_two_signals(wav_bytes):
    audio = ap.load_audio(wav_bytes, target_sr=8000)
    hss, lss = ap.separate(audio.samples, audio.sample_rate)
    assert hss.shape == audio.samples.shape
    assert lss.shape == audio.samples.shape
    assert np.isfinite(hss).all() and np.isfinite(lss).all()


def test_estimate_rate_bpm_detects_heart_rate():
    sr = 8000
    t = np.arange(sr * 5) / sr
    pulses = (np.sin(2 * np.pi * 1.5 * t) > 0.5).astype(float)
    sig = np.sin(2 * np.pi * 60 * t) * pulses
    bpm = ap.estimate_rate_bpm(sig, sr, min_bpm=40, max_bpm=200)
    assert bpm is not None
    assert 80 <= bpm <= 100  # 1.5 Hz == 90 BPM


def test_band_filters_reduce_out_of_band_energy():
    sr = 8000
    t = np.arange(sr * 2) / sr
    low = np.sin(2 * np.pi * 50 * t)
    high = np.sin(2 * np.pi * 1000 * t)
    mixed = low + high
    bp = ap.kaiser_bpf(mixed, sr, 20, 200)
    assert ap.band_energy(bp, sr, 20, 200) > ap.band_energy(bp, sr, 500, 2000)


def test_snr_db_positive_for_clean_signal():
    sr = 8000
    sig = np.sin(2 * np.pi * 60 * np.arange(sr) / sr)
    noise = 0.01 * np.random.RandomState(1).randn(sr)
    assert ap.snr_db(sig, noise) > 20
