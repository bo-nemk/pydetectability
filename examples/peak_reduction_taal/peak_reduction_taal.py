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
print(np.max(x))

# Resample
sampling_rate = 1000
x = scipy.signal.resample_poly(x, sampling_rate, sampling_rate_original)

# Parameters
N_segment = int(0.020 * sampling_rate)
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

# Create taal model
N_filter = 64
taal_model = pp.models.taal_model(sampling_rate, N_segment, mapping, N_filters=N_filter, training_rate=200)
auditory_filters = np.fft.irfft(taal_model.auditory_filter_bank_freq, n=N_segment, axis=1)

y_time_bin  = cp.Variable(N_segment)
g_time_bin  = cp.Parameter((N_filter, N_segment))
gx_time_bin = cp.Parameter((N_filter, N_segment))

taal_problem = cp.Problem(cp.Minimize(cp.norm(y_time_bin, p='inf')), 
    constraints=[
        # NOTE: add circulant matrix? Slow as hell already
        sum([cp.power(cp.norm(cp.multiply(g_time_bin[i,:], 
            sp.linalg.circulant(auditory_filters[i]) @ y_time_bin) - gx_time_bin[i,:]), 2.0) for i in range(0, N_filter)]) <= 2.0,
    ]
)

y_taal_stft_time = np.zeros(x_stft_time.shape)
for idx, pair in enumerate(zip(x_stft_freq.T, x_stft_time.T)):
    g_time_bin.value = taal_model.gain(pair[1])
    gx_time_bin.value = np.array([(sp.linalg.circulant(auditory_filters[i]) @ pair[1]) * g_time_bin.value[i] for i in range(0, N_filter)])

    taal_problem.solve(warm_start=True, verbose=False)

    y_taal_stft_time[:, idx] = y_time_bin.value
    print(f"idx :: {idx} / {x_stft_freq.shape[1]}")

y_taal = np.zeros(y_taal_stft_time.shape[0] + N_overlap * y_taal_stft_time.shape[1])
window_correction = np.zeros(y_taal.shape)

stft_indices = lb.frames_to_samples(range(0, y_taal_stft_time.shape[1]), N_segment - N_overlap)
for idx, index in enumerate(stft_indices):
    y_taal[index : index + N_segment] += window_hann * y_taal_stft_time[:, idx]
    window_correction[index : index + N_segment] += window_hann

window_correction[window_correction == 0] = 1.0
y_taal = y_taal / window_correction
sp.io.wavfile.write("output_taal.wav", sampling_rate_original, np.rint(sp.signal.resample_poly(y_taal, sampling_rate_original,
    sampling_rate)).astype(np.int16))

x_pad = np.pad(x,(0, max(y_taal.size - x.size, x.size - y_taal.size)))
d_taal =  x_pad - y_taal 
sp.io.wavfile.write("output_taal_d.wav", sampling_rate_original, np.rint(sp.signal.resample_poly(d_taal, sampling_rate_original,
    sampling_rate)).astype(np.int16))
sp.io.wavfile.write("reference.wav", sampling_rate_original, np.rint(sp.signal.resample_poly(x_pad, sampling_rate_original,
    sampling_rate)).astype(np.int16))
