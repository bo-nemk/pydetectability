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
sampling_rate, x = sp.io.wavfile.read("input_1.wav")
x = x[0 : sampling_rate * 3]

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
mapping = pp.utility.signal_pressure_mapping(1, 70)

# Create par model and taal model
par_model = pp.models.par_model(sampling_rate, N_segment, mapping)

a = cp.Variable(N_filters)
gn_freq_bin = cp.Parameter((N_filters, x_stft_freq.shape[0]), complex=True)

par_problem = cp.Problem(cp.Maximize(cp.sum(a)), 
    constraints=[
        cp.power(cp.norm(sum([a[i] * gn_freq_bin[i,:] for i in range(0, N_filters)])), 2.0) <= 1.0,
    ]
)

y_par_stft_freq  = np.zeros(x_stft_freq.shape).astype(complex)
for idx in range(0, N_frames):
    gn_freq_bin.value = np.array([n_stft_freq[i][:, idx] * par_model.gain(x_stft_time[:, idx]) for i in range(0, N_filters)])
    par_problem.solve(warm_start=True, verbose=True)
    y_par_stft_freq[:, idx] = x_stft_freq[:, idx] + sum([n_stft_freq[i][:, idx] * a.value[i] for
        i in range(0, N_filters)])
    print(f"idx :: {idx} / {x_stft_freq.shape[1]}")
    print(f"a :: {a.value}")

y_par_stft_time = np.fft.irfft(y_par_stft_freq, n=N_segment, axis=0)
y_par = np.zeros(y_par_stft_time.shape[0] + N_overlap * y_par_stft_time.shape[1])

stft_indices = lb.frames_to_samples(range(0, y_par_stft_time.shape[1]), N_segment - N_overlap)
for idx, index in enumerate(stft_indices):
    y_par[index : index + N_segment] += window_hann * y_par_stft_time[:, idx]
sp.io.wavfile.write("output_par.wav", sampling_rate, np.rint(y_par).astype(np.int16))

x_pad = np.pad(x,(0, max(y_par.size - x.size, x.size - y_par.size)))
d_par =  x_pad - y_par 
sp.io.wavfile.write("output_par_d.wav", sampling_rate, np.rint(d_par).astype(np.int16))
