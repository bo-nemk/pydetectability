#!/bin/python3
import pydetectability as pp
import cvxpy as cp
import numpy as np
import scipy as sp 
import scipy.io.wavfile
import scipy.linalg
import scipy.signal
import matplotlib.pyplot as mpl
import librosa as lb

# Import song 
sampling_rate, x = sp.io.wavfile.read("../input_1.wav")
x = x[0 : sampling_rate * 3, 0]

# Generate noise
N_filters = 100 
n = np.zeros((N_filters, x.size))
bandwidth = 0.8 * (sampling_rate / 2) / N_filters
for idx, f in enumerate(np.linspace(0.1 * (sampling_rate / 2), 0.6* (sampling_rate / 2), N_filters)):
    noise = np.random.normal(0, 1, x.size)
    sos = sp.signal.butter(8, [f - bandwidth / 2, f + bandwidth / 2], btype='bandpass', output='sos', fs=sampling_rate)
    n[idx, :] = sp.signal.sosfilt(sos, noise)[0 : x.size]

# Parameters
N_segment = int(0.02 * sampling_rate)
print(f"N_segment: {N_segment}")
N_overlap = int(N_segment / 2)
print(f"N_overlap: {N_overlap}")

# Windows
window_hann = sp.signal.windows.hann(N_segment)

# Perform windowing through default functions
x_stft_time = lb.util.frame(x, N_segment, N_segment - N_overlap)
x_stft_freq = np.fft.rfft(x_stft_time, n=N_segment, axis=0)
N_frames = x_stft_time.shape[1]

n_stft_time = [lb.util.frame(n[i], N_segment, N_segment - N_overlap) for i in range(0, N_filters)]
n_stft_freq = [np.fft.rfft(n_stft_time[i], n=N_segment, axis=0) for i in range(0, N_filters)]

# Create default mapping
mapping = pp.utility.signal_pressure_mapping(1, 100)

# Create taal model
taal_model = pp.models.taal_model(sampling_rate, N_segment, mapping)
auditory_filters = np.fft.irfft(taal_model.auditory_filter_bank_freq, n=N_segment, axis=1)

y_time_bin  = cp.Variable(x_stft_time.shape[0])
x_time_bin  = cp.Parameter(x_stft_time.shape[0])
g_time_bin  = cp.Parameter((x_stft_time.shape[0], 64))
ghx_time_bin = cp.Parameter((x_stft_time.shape[0], 64))

taal_problem = cp.Problem(cp.Minimize(cp.norm(y_time_bin, p='inf')), 
    constraints=[
        sum([cp.power(cp.norm(cp.multiply(g_time_bin[:,i], sp.linalg.circulant(auditory_filters[i]) @ y_time_bin) -
            ghx_time_bin[:,i]), 2.0) for i in range(0, 64)]) <= 1.0
    ]
)

y_taal_stft_time = np.zeros(x_stft_time.shape)
for idx, pair in enumerate(zip(x_stft_freq.T, x_stft_time.T)):
    x_time_bin.value = pair[1]
    g_time_bin.value = taal_model.gain(x_time_bin.value).T
    ghx_time_bin.value = [g_time_bin.value[:,i] * (sp.linalg.circulant(auditory_filters[i]) @ 
        x_time_bin.value) for i in range(0, 64)]

    taal_problem.solve(warm_start=True, verbose=True)

    y_taal_stft_time[:, idx] = y_time_bin.value
    print(f"idx :: {idx} / {x_stft_time.shape[1]}")

y_taal = np.zeros(y_taal_stft_time.shape[0] + N_overlap * y_taal_stft_time.shape[1])

stft_indices = lb.frames_to_samples(range(0, y_taal_stft_time.shape[1]), N_segment - N_overlap)
for idx, index in enumerate(stft_indices):
    y_taal[index : index + N_segment] += window_hann * y_taal_stft_time[:, idx]
sp.io.wavfile.write("output_taal.wav", sampling_rate, np.rint(y_taal).astype(np.int16))

x_pad = np.pad(x,(0, max(y_taal.size - x.size, x.size - y_taal.size)))
d_taal =  x_pad - y_taal 
sp.io.wavfile.write("output_taal_d.wav", sampling_rate, np.rint(d_taal).astype(np.int16))
