#!/bin/python3
import pydetectability as pp

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

# Parameters
N_segment = int(0.2 * sampling_rate)
print(f"N_segment: {N_segment}")
N_overlap = int(N_segment / 2)
print(f"N_overlap: {N_overlap}")

# Windows
window_hann = sp.signal.windows.hann(N_segment)
window_rect = sp.signal.windows.boxcar(N_segment)

# Perform windowing through default functions
freq, time, x_stft_freq = sp.signal.stft(x, sampling_rate, nperseg=N_segment, noverlap=N_overlap, window=window_hann)
x_stft_time = np.fft.irfft(x_stft_freq, n=N_segment, axis=0)
print(f"x_stft_time: {x_stft_time.shape}")
print(f"x_stft_freq: {x_stft_freq.shape}")
# Create default mapping
mapping = pp.utility.signal_pressure_mapping(1, 100)

x_freq_bin = cp.Parameter(x_stft_freq.shape[0], complex=True)
x_time_bin = cp.Parameter(x_stft_time.shape[0])
y_freq_bin = cp.Variable(x_stft_freq.shape[0], complex=True)
y_time_bin = cp.Variable(x_stft_time.shape[0])
g_freq_bin = cp.Parameter(x_stft_freq.shape[0], complex=True)
g_time_bin = cp.Parameter(x_stft_freq.shape[0])

# Create par model and taal model
par_model = pp.models.par_model_cvx(sampling_rate, N_segment, mapping)
par_problem = cp.Problem(cp.Minimize(cp.norm(y_freq_bin, p='inf')), 
    constraints=[
        par_model.detectability_gain_cvx(g_freq_bin, x_freq_bin - y_freq_bin) <= 1,
    ]
)

y_par_stft_freq  = np.zeros(x_stft_freq.shape).astype(complex)
y_taal_stft_freq = np.zeros(x_stft_freq.shape).astype(complex)

for idx, pair in enumerate(zip(x_stft_freq.T, x_stft_time.T)):
    x_freq_bin.value = pair[0]
    x_time_bin.value = pair[1]

    g_freq_bin.value = par_model.gain(x_time_bin.value)

    par_problem.solve(verbose=True)

    y_par_stft_freq[:, idx] = y_freq_bin.value
    print(f"y: {np.linalg.norm(y_freq_bin.value)}, x:{np.linalg.norm(x_freq_bin.value)}")

time, y_par = scipy.signal.istft(y_par_stft_freq, sampling_rate, nperseg=N_segment, noverlap=N_overlap, window=window_hann)
time, y_taal = scipy.signal.istft(y_taal_stft_freq, sampling_rate, nperseg=N_segment, noverlap=N_overlap, window=window_hann)

sp.io.wavfile.write("output_par.wav", sampling_rate, y_par)
sp.io.wavfile.write("output_taal.wav", sampling_rate, y_taal)
