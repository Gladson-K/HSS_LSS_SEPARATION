#!/usr/bin/env python3
"""
Wavelet feature extraction (like existing baseline)
For feature-based input approach
"""
import numpy as np
import pywt
import librosa


def compute_wavelet_features(X, a, b, sr=9000, wavelet='morl', frame_size=400):
    """Compute continuous wavelet transform features"""
    widths = np.arange(1, 256)
    dt = 1/sr
    frequencies = pywt.scale2frequency(wavelet, widths) / dt
    
    # Filter frequencies
    upper_idx = ([x for x in range(len(widths)) if frequencies[x] > b])[-1] if any(frequencies[x] > b for x in range(len(widths))) else len(widths)-1
    lower_idx = ([x for x in range(len(widths)) if frequencies[x] < a])[0] if any(frequencies[x] < a for x in range(len(widths))) else 0
    
    widths_hss = widths[upper_idx:lower_idx] if upper_idx < lower_idx else widths
    
    wavelet_coeffs_hss, freqs_hss = pywt.cwt(X, widths_hss, wavelet=wavelet, sampling_period=dt)
    
    # Split into frames
    start = 0
    end_hss = wavelet_coeffs_hss.shape[1]
    frames_hss = []
    
    while start + frame_size <= end_hss - 1:
        f_hss = wavelet_coeffs_hss[:, start:start+frame_size]
        frames_hss.append(f_hss)
        start += frame_size
    
    if len(frames_hss) == 0:
        return np.array([])
    
    frames_hss = np.array(frames_hss)
    frames_hss = frames_hss.reshape((len(frames_hss), wavelet_coeffs_hss.shape[0], frame_size))
    return frames_hss
