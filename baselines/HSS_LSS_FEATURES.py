
import numpy as np
# import matplotlib.pyplot as plt
# import datetime
import scipy.fftpack
# import tensorflow as tf
# from tensorflow import keras

import os
import glob
import librosa


import scipy.signal
import scipy.io.wavfile
import pywt
from skimage.restoration import denoise_wavelet
import pandas as pd

def cwt_rec(times, filtered_hss_array, widths):
    # Define your wavelet function, for example, the Ricker wavelet
    def ricker(x, a):
        return ((1 - (x - 6 )*2) / a* 2) * np.exp(-(x - 6) * 2 / (2 * a * 2))

    # Create a 1D grid of values for the wavelet
    t = np.linspace(-6, 6, len(filtered_hss_array), endpoint=False)

    # Initialize an array to store the wavelet coefficients
    filtered_hss_cwt = np.zeros((len(widths), len(t)))

    # Apply the wavelet transform for each scale
    for i in enumerate(widths):
        # Generate the wavelet
        wavelet_data = ricker(t, 6)

        # Perform the convolution
        filtered_hss_cwt[i, :] = np.convolve(filtered_hss_array, wavelet_data, mode='same')

    filtered_hss_cwt = denoise_wavelet(filtered_hss_cwt, wavelet='db1', mode='soft', wavelet_levels=4,
                                       method='BayesShrink', rescale_sigma=True)

    coeff_hss = pywt.wavedec(filtered_hss_cwt, 'db2', 'smooth', level=3)
    coeff_hss_rec = pywt.waverec(coeff_hss, 'db2')

    return coeff_hss_rec, widths


# Rest of your code remains the same

# Example usage:
# widths3 = np.arange(1, num_scales + 1)
# filtered_hss_rec, widths3 = cwt_rec(times1, filtered_hss, widths3)


def hss_S1(filtered_hss):
    with open(file_path, 'rb') as filtered_hss:
        sample_data, data2 = scipy.io.wavfile.read(filtered_hss)
        times3 = np.arange(len(data2)) / sample_rate

    filtered_hss_S1 = bandpass(data2, [20, 60], sample_rate)

    return filtered_hss_S1, times3


def hss_S2(filtered_hss):
    with open(file_path, 'rb') as filtered_hss:
        sample_data, data2 = scipy.io.wavfile.read(filtered_hss)
        times3 = np.arange(len(data2)) / sample_rate

    filtered_hss_S2 = bandpass(data2, [80, 200], sample_rate)

    return filtered_hss_S2, times3


def hss_S3(filtered_hss):
    with open(file_path, 'rb') as filtered_hss:
        sample_data, data2 = scipy.io.wavfile.read(filtered_hss)
        times3 = np.arange(len(data2)) / sample_rate

    filtered_hss_S3 = lowpass(data2, 40, sample_rate)

    return filtered_hss_S3, times3


def hss_S4(filtered_hss):
    with open(file_path, 'rb') as filtered_hss:
        sample_data, data2 = scipy.io.wavfile.read(filtered_hss)
        times3 = np.arange(len(data2)) / sample_rate

    filtered_hss_S4 = bandpass(data2, [20, 40], sample_rate)

    return filtered_hss_S4, times3


def lss_vesicular(filtered_lss):
    with open(file_path, 'rb') as filtered_lss:
        sample_data, data3 = scipy.io.wavfile.read(filtered_lss)
        times4 = np.arange(len(data3)) / sample_rate

    filtered_lss_vesicular = lowpass(data3, 800, sample_rate)

    return filtered_lss_vesicular, times4


def lss_bronchial(filtered_lss):
    with open(file_path, 'rb') as filtered_lss:
        sample_data, data3 = scipy.io.wavfile.read(filtered_lss)
        times4 = np.arange(len(data3)) / sample_rate

    filtered_lss_bronchial = bandpass(data3, [150, 3000], sample_rate)

    return filtered_lss_bronchial, times4


def lss_crackles_fine(filtered_lss):
    with open(file_path, 'rb') as filtered_lss:
        sample_data, data3 = scipy.io.wavfile.read(filtered_lss)
        times4 = np.arange(len(data3)) / sample_rate

    filtered_lss_crackles_fine = bandpass(data3, [400, 2000], sample_rate)

    return filtered_lss_crackles_fine, times4


def lss_crackles_coarse(filtered_lss):
    with open(file_path, 'rb') as filtered_lss:
        sample_data, data3 = scipy.io.wavfile.read(filtered_lss)
        times4 = np.arange(len(data3)) / sample_rate

    filtered_lss_crackles_coarse = bandpass(data3, [100, 600], sample_rate)

    return filtered_lss_crackles_coarse, times4


def lss_wheezes(filtered_lss):
    with open(file_path, 'rb') as filtered_lss:
        sample_data, data3 = scipy.io.wavfile.read(filtered_lss)
        times4 = np.arange(len(data3)) / sample_rate

    filtered_lss_wheezes = bandpass(data3, [100, 1000], sample_rate)

    return filtered_lss_wheezes, times4


# Call the hss_features function to get the features and times


def lowpass(data: np.ndarray, cutoff: float, sample_rate: float, poles: int = 5):
    sos = scipy.signal.butter(poles, cutoff, 'lowpass', fs=sample_rate, output='sos')
    filtered_data = scipy.signal.sosfiltfilt(sos, data)
    return filtered_data


def bandpass(data: np.ndarray, edges: list[float], sample_rate: float, poles: int = 5):
    sos = scipy.signal.butter(poles, edges, 'bandpass', fs=sample_rate, output='sos')
    filtered_data = scipy.signal.sosfiltfilt(sos, data)
    return filtered_data


folder_path = 'D:\Heart Lung Sound Separation_SIG\Data Sets\DATA';
wav_files = glob.glob(os.path.join(folder_path, '*.wav'))

audio_data = []  # to store the loaded audio data

filtered_hss = []
filtered_hss_S1 = []
filtered_hss_S2 = []
filtered_hss_S3 = []
filtered_hss_S4 = []

filtered_lss = []
filtered_lss_vesicular = []
filtered_lss_bronchial = []
filtered_lss_crackles_fine = []
filtered_lss_crakles_coarse = []
filtered_lss_wheezes = []

filtered_hss_rec = []
filtered_hss_rec_S1 = []
filtered_hss_rec_S2 = []
filtered_hss_rec_S3 = []
filtered_hss_rec_S4 = []

filtered_lss_rec = []
filtered_lss_rec_vesicular = []
filtered_lss_rec_bronchial = []
filtered_lss_rec_crackles_fine = []
filtered_lss_rec_crakles_coarse = []
filtered_lss_rec_wheezes = []

for file_path in wav_files:
    # Load the audio file using librosa
    audio, sr = librosa.load(file_path, sr=None)  # sr=None means librosa will use the native sample rate

    # Append the audio data to the list
    audio_data.append(audio)

# Load sample data from a WAV file
with open(file_path, 'rb') as audio:
    sample_rate, data = scipy.io.wavfile.read(audio)
    times1 = np.arange(len(data)) / sample_rate

    # Apply a 10-50 Hz high-pass filter to the original data
    filtered_hss = bandpass(data, [20, 300], sample_rate)

# Load sample data from a WAV file
with open(file_path, 'rb') as audio:
    sample_rate, data1 = scipy.io.wavfile.read(audio)
    times2 = np.arange(len(data1)) / sample_rate

    filtered_lss = lowpass(data1, 4200, sample_rate)

    filtered_hss_S1 = hss_S1(filtered_hss)

    filtered_hss_S2 = hss_S2(filtered_hss)

    filtered_hss_S3 = hss_S3(filtered_hss)

    filtered_hss_S4 = hss_S4(filtered_hss)

    filtered_lss_vesicular = lss_vesicular(filtered_lss)
    filtered_lss_bronchial = lss_bronchial(filtered_lss)
    filtered_lss_crackles_fine = lss_crackles_fine(filtered_lss)
    filtered_lss_crackles_coarse = lss_crackles_coarse(filtered_lss)
    filtered_lss_wheezes = lss_wheezes(filtered_lss)

    num_scales = 100
    widths3 = np.arange(1, num_scales + 1)

    filtered_hss_rec, widths3 = cwt_rec(times1, filtered_hss, widths3)
    filtered_lss_rec, widths4 = cwt_rec(times2, filtered_lss, widths3)

    filtered_hss_rec_S1 = hss_S1(filtered_hss_rec)

    filtered_hss_rec_S2 = hss_S2(filtered_hss_rec)

    filtered_hss_rec_S3 = hss_S3(filtered_hss_rec)

    filtered_hss_rec_S4 = hss_S4(filtered_hss_rec)

    filtered_lss_rec_vesicular = lss_vesicular(filtered_lss_rec)
    filtered_lss_rec_bronchial = lss_bronchial(filtered_lss_rec)
    filtered_lss_rec_crackles_fine = lss_crackles_fine(filtered_lss_rec)
    filtered_lss_rec_crackles_coarse = lss_crackles_coarse(filtered_lss_rec)
    filtered_lss_rec_wheezes = lss_wheezes(filtered_lss_rec)

    filtered_hss.append(filtered_hss)
    filtered_hss_S1.append(filtered_hss_S1)
    filtered_hss_S2.append(filtered_hss_S2)
    filtered_hss_S3.append(filtered_hss_S3)
    filtered_hss_S4.append(filtered_hss_S4)

    filtered_lss.append(filtered_lss)
    filtered_lss_vesicular.append(filtered_lss_vesicular)
    filtered_lss_bronchial.append(filtered_lss_bronchial)
    filtered_lss_crackles_fine.append(filtered_lss_crackles_fine)
    filtered_lss_crackles_coarse.append(filtered_lss_crackles_coarse)
    filtered_lss_wheezes.append(filtered_lss_wheezes)

    filtered_hss_rec.append(filtered_hss_rec)
    filtered_lss_rec.append(filtered_lss_rec)

    filtered_hss_S1.append(filtered_hss_S1)
    filtered_hss_S2.append(filtered_hss_S2)
    filtered_hss_S3.append(filtered_hss_S3)
    filtered_hss_S4.append(filtered_hss_S4)

    filtered_lss_rec_vesicular.append(filtered_lss_vesicular)
    filtered_lss_rec_bronchial.append(filtered_lss_bronchial)
    filtered_lss_rec_crackles_fine.append(filtered_lss_crackles_fine)
    filtered_lss_rec_crackles_coarse.append(filtered_lss_crackles_coarse)
    filtered_lss_rec_wheezes.append(filtered_lss_wheezes)

# Create a dictionary with the data
data_dict = {
    'audio_data': audio_data,
    'filtered_hss': filtered_hss,
    'filtered_hss_S1': filtered_hss_S1,
    'filtered_hss_S2': filtered_hss_S2,
    'filtered_hss_S3': filtered_hss_S3,
    'filtered_hss_S4': filtered_hss_S4,
    'filtered_lss': filtered_lss,
    'filtered_lss_vesicular': filtered_lss_vesicular,
    'filtered_lss_bronchial': filtered_lss_bronchial,
    'filtered_lss_crackles_fine': filtered_lss_crackles_fine,
    'filtered_lss_rec_crackles_coarse': filtered_lss_rec_crackles_coarse,
    'filtered_lss_wheezes': filtered_lss_wheezes,
    'filtered_hss_rec': filtered_hss_rec,
    'filtered_hss_rec_S1': filtered_hss_rec_S1,
    'filtered_hss_rec_S2': filtered_hss_rec_S2,
    'filtered_hss_rec_S3': filtered_hss_rec_S3,
    'filtered_hss_rec_S4': filtered_hss_rec_S4,
    'filtered_lss_rec': filtered_lss_rec,
    'filtered_lss_rec_vesicular': filtered_lss_vesicular,
    'filtered_lss_rec_bronchial': filtered_lss_bronchial,
    'filtered_lss_rec_crackles_fine': filtered_lss_crackles_fine,
    'filtered_lss_rec_crackles_coarse': filtered_lss_rec_crackles_coarse,
    'filtered_lss_rec_wheezes': filtered_lss_wheezes,

}

# Convert the dictionary to a DataFrame
df = pd.DataFrame(data_dict)

# Save the DataFrame to a CSV file
df.to_csv('HSS_LSS_Separated_data_list.csv', index=False)








