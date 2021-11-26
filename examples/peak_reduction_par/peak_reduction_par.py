#!/bin/python3
import pydetectability as pp
import cvxpy as cp
import numpy as np
import scipy as sp 
import scipy.io.wavfile
import scipy.signal
import matplotlib.pyplot as mpl
import librosa as lb

# Import song 
sampling_rate_original, x = sp.io.wavfile.read("../kickdrum.wav")
x = x[:, 0]

# Resample
sampling_rate = 1000
x = scipy.signal.resample_poly(x, sampling_rate, sampling_rate_original)

# Parameters
N_segment = int(0.02 * sampling_rate)
print(f"N_segment: {N_segment}")
N_overlap = int(N_segment / 2)
print(f"N_overlap: {N_overlap}")

# Windows
window_hann = sp.signal.windows.hann(N_segment, sym=False)

# Perform windowing through default functions
x_stft_time = lb.util.frame(x, N_segment, N_segment - N_overlap)
x_stft_freq = np.fft.rfft(x_stft_time, n=N_segment, axis=0)

# Create default mapping
mapping = pp.utility.signal_pressure_mapping(1, 100)

# Create par model and taal model
par_model = pp.models.par_model(sampling_rate, N_segment, mapping, training_rate = 200)

y_freq_bin  = cp.Variable(x_stft_freq.shape[0], complex=True)
g_freq_bin  = cp.Parameter(x_stft_freq.shape[0], complex=True)
gx_freq_bin = cp.Parameter(x_stft_freq.shape[0], complex=True)

detectability = cp.power(cp.norm(cp.multiply(g_freq_bin, y_freq_bin) - gx_freq_bin), 2.0)
constraints = [detectability <= 2.0]
par_problem = cp.Problem(cp.Minimize(cp.norm(y_freq_bin, p='inf')), constraints=constraints)

y_par_stft_freq  = np.zeros(x_stft_freq.shape).astype(complex)
for idx, pair in enumerate(zip(x_stft_freq.T, x_stft_time.T)):
    g_freq_bin.value = par_model.gain(pair[1])
    gx_freq_bin.value = pair[0] * g_freq_bin.value
    print(gx_freq_bin.value)
    print(g_freq_bin.value)

    par_problem.solve(warm_start=True)
    print(detectability.value)

    y_par_stft_freq[:, idx] = y_freq_bin.value
    print(f"idx :: {idx} / {x_stft_freq.shape[1]}")

y_par_stft_time = np.fft.irfft(y_par_stft_freq, n=N_segment, axis=0)
y_par = np.zeros(y_par_stft_time.shape[0] + N_overlap * y_par_stft_time.shape[1])
window_correction = np.zeros(y_par.shape)

stft_indices = lb.frames_to_samples(range(0, y_par_stft_time.shape[1]), N_segment - N_overlap)
for idx, index in enumerate(stft_indices):
    y_par[index : index + N_segment] += window_hann * y_par_stft_time[:, idx]
    window_correction[index : index + N_segment] += window_hann

window_correction[window_correction == 0] = 1.0
y_par = y_par / window_correction
sp.io.wavfile.write("output_par.wav", sampling_rate_original, np.rint(sp.signal.resample_poly(y_par, sampling_rate_original,
    sampling_rate)).astype(np.int16))

x_pad = np.pad(x,(0, max(y_par.size - x.size, x.size - y_par.size)))
d_par =  x_pad - y_par 
sp.io.wavfile.write("output_par_d.wav", sampling_rate_original, np.rint(sp.signal.resample_poly(d_par, sampling_rate_original,
    sampling_rate)).astype(np.int16))
sp.io.wavfile.write("reference.wav", sampling_rate_original, np.rint(sp.signal.resample_poly(x_pad, sampling_rate_original,
    sampling_rate)).astype(np.int16))
