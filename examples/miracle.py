#!/bin/python3
# Assumes lib is installed import pydetectability as pp Implements 13db miracle through an optimization problem
import cvxpy as cp

# Import science libs 
import numpy as np
import scipy as sp 
import scipy.io.wavfile
import scipy.signal
import matplotlib.pyplot as mpl

# Import song as mono
sampling_rate, x = sp.io.wavfile.read("input.wav")
x = x[:, 0]

# Create some noise equal to song length
n = np.random.normal(0, 1, x.size)

# Parameters
N_segment = int(0.02 * sampling_rate)
N_overlap = int(N_segment / 2)

# Windows
window_hann = sp.signal.windows.hann(N_segment)
window_rect = sp.signal.windows.boxcar(N_segment)

# Perform windowing through default functions
freq, time, x_stft_freq = sp.signal.stft(x, sampling_rate, nperseg=N_segment, noverlap=N_overlap, window=window_hann)
x_stft_time = np.fft.irfft(x_stft_freq, n=N_segment, axis=0)

freq, time, n_stft_freq = sp.signal.stft(n, sampling_rate, nperseg=N_segment, noverlap=N_overlap, window=window_hann)
n_stft_time = np.fft.irfft(n_stft_freq, n=N_segment, axis=0)

# Create default mapping
mapping = pp.utility.signal_pressure_mapping(1, 100)

a = cp.Variable(1)
x_freq_bin = cp.Parameter(x_stft_freq.shape[1], complex=True)
x_time_bin = cp.Parameter(x_stft_time.shape[1])
n_freq_bin = cp.Parameter(x_stft_freq.shape[1], complex=True)
n_time_bin = cp.Parameter(x_stft_time.shape[1])

# Create par model and define problem
par_model = pp.models.par_model_cvx(sampling_rate, N_segment, mapping)
par_problem = cp.Problem(cp.Minimize(par_model.detectability_gain(x_stft_freq, x_freq_bin + a * n_freq_bin)))

y_stft_freq = np.zeros(x_stft_freq.shape).astype(complex)
for idx, pair in enumerate(zip(x_stft_freq, n_stft_freq)):
    x_freq_bin = pair[0]
    n_freq_bin = pair[1]

    y_freq_bin.value = x_freq_bin
    y_stft_freq[idx, :] = y_freq_bin.value

time, y_par = scipy.signal.istft(y_stft_freq, sampling_rate, nperseg=N_segment, noverlap=N_overlap, window=window_hann)

print(x[0:100])
print(np.rint(y_par[0:100]).astype(int))
