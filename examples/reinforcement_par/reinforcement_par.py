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
sampling_rate, x_1 = sp.io.wavfile.read("../input_1.wav")
x_1 = x_1[0 :, 0]

sampling_rate, x_2 = sp.io.wavfile.read("../input_2.wav")
x_2 = 0.2 * x_2[0 :, 0]

# Parameters
N_segment = int(0.02 * sampling_rate)
print(f"N_segment: {N_segment}")
N_overlap = int(N_segment / 2)
print(f"N_overlap: {N_overlap}")

# Windows
window_hann = sp.signal.windows.hann(N_segment)

# Perform windowing through default functions
x_1_stft_time = lb.util.frame(x_1, N_segment, N_segment - N_overlap)
x_1_stft_freq = np.fft.rfft(x_1_stft_time, n=N_segment, axis=0)

x_2_stft_time = lb.util.frame(x_2, N_segment, N_segment - N_overlap)
x_2_stft_freq = np.fft.rfft(x_2_stft_time, n=N_segment, axis=0)

N_frames = x_1_stft_time.shape[1]

# Create default mapping
mapping = pp.utility.signal_pressure_mapping(1, 70)

# Create par model and taal model
par_model = pp.models.par_model(sampling_rate, N_segment, mapping)

r_freq = cp.Variable(x_1_stft_freq.shape[0], complex=True)

g_freq = cp.Parameter(x_1_stft_freq.shape[0], complex=True)
n_freq = cp.Parameter(x_1_stft_freq.shape[0], complex=True)

par_problem = cp.Problem(cp.Minimize(cp.power(cp.norm(cp.multiply(g_freq,  r_freq) + n_freq), 2.0)), 
    constraints=[
        cp.power(cp.norm(cp.multiply(g_freq, r_freq)), 2.0) <= 5.0,
    ]
)

y_par_stft_freq  = np.zeros(x_1_stft_freq.shape).astype(complex)
for idx in range(0, N_frames):
    g_freq.value = par_model.gain(x_1_stft_time[:, idx])
    n_freq.value = g_freq.value * x_2_stft_freq[:, idx]
    par_problem.solve(warm_start=True, verbose=False)

    y_par_stft_freq[:, idx] = x_1_stft_freq[:, idx] + r_freq.value 
    print(f"idx :: {idx + 1} / {N_frames}")

y_par_stft_time = np.fft.irfft(y_par_stft_freq, n=N_segment, axis=0)
y_par = np.zeros(y_par_stft_time.shape[0] + N_overlap * y_par_stft_time.shape[1])

stft_indices = lb.frames_to_samples(range(0, y_par_stft_time.shape[1]), N_segment - N_overlap)
for idx, index in enumerate(stft_indices):
    y_par[index : index + N_segment] += window_hann * y_par_stft_time[:, idx]
sp.io.wavfile.write("output.wav", sampling_rate, np.rint(y_par).astype(np.int16))

x_1_pad = np.pad(x_1,(0, max(y_par.size - x_1.size, x_1.size - y_par.size)))
x_2_pad = np.pad(x_2,(0, max(y_par.size - x_2.size, x_2.size - y_par.size)))
d_1_par =  x_1_pad - y_par 
sp.io.wavfile.write("output_difference.wav", sampling_rate, np.rint(d_1_par).astype(np.int16))
sp.io.wavfile.write("input_1.wav", sampling_rate, np.rint(x_1_pad).astype(np.int16))
sp.io.wavfile.write("input_2.wav", sampling_rate, np.rint(x_2_pad).astype(np.int16))

sp.io.wavfile.write("combined.wav", sampling_rate, np.rint(x_2_pad + y_par).astype(np.int16))
sp.io.wavfile.write("combined_reference.wav", sampling_rate, np.rint(x_2_pad + x_1_pad).astype(np.int16))
