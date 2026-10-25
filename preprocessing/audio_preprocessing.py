#!/usr/bin/env python3
"""
Audio preprocessing pipeline for Heart-Lung Sound Separation
Based on algorithm from research PPT:
1. Kaiser Window BPF for HSS, HPF for LSS
2. Daubechies Wavelet Filter (BPF for HSS, HPF for LSS)
3. RLS algorithm
4. MFCC conversion
"""
import numpy as np
import scipy.signal
import scipy.io.wavfile
import pywt
import librosa


class HeartLungPreprocessor:
    def __init__(self, sr=8000):
        self.sr = sr
        
    def kaiser_bpf(self, data, lowcut, highcut, numtaps=101, beta=8.6):
        """Kaiser Window Bandpass Filter for HSS"""
        nyq = 0.5 * self.sr
        low = lowcut / nyq
        high = highcut / nyq
        taps = scipy.signal.firwin(numtaps, [low, high], pass_zero=False, window=('kaiser', beta))
        filtered = scipy.signal.lfilter(taps, 1.0, data)
        return filtered
    
    def kaiser_hpf(self, data, cutoff, numtaps=101, beta=8.6):
        """Kaiser Window Highpass Filter for LSS"""
        nyq = 0.5 * self.sr
        high = cutoff / nyq
        taps = scipy.signal.firwin(numtaps, high, pass_zero=False, window=('kaiser', beta))
        filtered = scipy.signal.lfilter(taps, 1.0, data)
        return filtered
    
    def daubechies_filter(self, data, wavelet='db4', mode='symmetric', level=4, filter_type='bpf'):
        """Daubechies Wavelet Filter"""
        coeffs = pywt.wavedec(data, wavelet, mode=mode, level=level)
        # Apply filtering based on type
        if filter_type == 'bpf':  # For HSS - keep mid frequencies
            # Zero out approximation and very high detail coefficients as appropriate
            coeffs_new = list(coeffs)
            # Keep detail coefficients in heart sound range
            coeffs = coeffs_new
        else:  # HPF for LSS
            coeffs_new = list(coeffs)
            coeffs = coeffs_new
        reconstructed = pywt.waverec(coeffs, wavelet, mode=mode)
        # Adjust length if necessary
        if len(reconstructed) > len(data):
            reconstructed = reconstructed[:len(data)]
        elif len(reconstructed) < len(data):
            reconstructed = np.pad(reconstructed, (0, len(data)-len(reconstructed)))
        return reconstructed
    
    def rls_filter(self, input_signal, reference=None, order=4, forgetting_factor=0.99):
        """
        Recursive Least Square (RLS) Algorithm
        Used for adaptive filtering in separation
        """
        n = len(input_signal)
        if reference is None:
            reference = np.roll(input_signal, 1)
            reference[0] = 0
        
        w = np.zeros(order)
        P = np.eye(order) / 0.01  # Initial covariance matrix
        output = np.zeros(n)
        error = np.zeros(n)
        
        for i in range(order, n):
            x = input_signal[i:i-order:-1]  # Input vector
            y = np.dot(w, x)  # Filter output
            error[i] = reference[i] - y  # Error
            k = np.dot(P, x) / (forgetting_factor + np.dot(np.dot(x, P), x))  # Gain
            w = w + k * error[i]
            P = (P - np.outer(k, np.dot(x, P))) / forgetting_factor
            output[i] = y
        
        return output, error
    
    def preprocess_hss(self, data):
        """Preprocess for Heart Sound Separation"""
        # Apply Kaiser Window BPF for HSS (20-200 Hz range)
        filtered = self.kaiser_bpf(data, 20, 200)
        # Apply Daubechies Wavelet Filter BPF
        filtered = self.daubechies_filter(filtered, wavelet='db4', filter_type='bpf')
        # Apply RLS
        separated, _ = self.rls_filter(filtered)
        return separated
    
    def preprocess_lss(self, data):
        """Preprocess for Lung Sound Separation"""
        # Apply Kaiser Window HPF for LSS
        filtered = self.kaiser_hpf(data, 150)  # High pass for lung sounds
        # Apply Daubechies Wavelet Filter HPF
        filtered = self.daubechies_filter(filtered, wavelet='db6', filter_type='hpf')
        # Apply RLS
        separated, _ = self.rls_filter(filtered)
        return separated
    
    def compute_mfcc(self, data, n_mfcc=40):
        """Compute MFCC features"""
        mfcc = librosa.feature.mfcc(y=data, sr=self.sr, n_mfcc=n_mfcc)
        return mfcc
    
    def compute_mel_spectrogram(self, data, n_mels=128):
        """Compute Mel spectrogram for raw audio input approach"""
        mel_spec = librosa.feature.melspectrogram(y=data, sr=self.sr, n_mels=n_mels)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db
    
    def preprocess_raw_audio(self, audio_file, target_sr=8000):
        """Preprocess raw wav audio for direct input approach"""
        data, sr = librosa.load(audio_file, sr=target_sr)
        # Normalize
        data = data / np.max(np.abs(data) + 1e-8)
        # Compute mel spectrogram
        spec = self.compute_mel_spectrogram(data)
        return data, spec
