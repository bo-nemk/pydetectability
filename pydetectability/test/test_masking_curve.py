import pytest 
import os
import matplotlib.pyplot as mpl
import numpy as np
import scipy as sp 
import scipy.signal.windows

from pydetectability.models.taal_model import taal_model
from pydetectability.models.par_model import par_model
from pydetectability.utility.signal_pressure_mapping import signal_pressure_mapping
from pydetectability.utility.threshold_in_quiet import threshold_in_quiet_db

def test_masking_curve():
    # Define problem parameters
    sampling_rate = 48000.0
    N_samples = int(0.2 * sampling_rate)
    N_filters = 64

    mapping = signal_pressure_mapping(1, 100)
    taal = taal_model(sampling_rate, N_samples, mapping, N_filters=N_filters)
    par = par_model(sampling_rate, N_samples, mapping, N_filters=N_filters)
    frequency = taal.frequency_axis


    window = sp.signal.windows.boxcar(N_samples)
    N_segment = int(0.05 * N_samples)
    test_ramp = np.concatenate((np.linspace(0, 1, int(0.1 * N_segment)), np.ones(N_segment - 2 * int(0.1 * N_segment)),
        np.linspace(1, 0, int(0.1 * N_segment))))
    test_sine = mapping.pressure_to_signal(32) * test_ramp * np.cos(2 * np.pi * 1000 * np.arange(0, N_segment) / sampling_rate)
    
    test_sine_signals = [window * np.pad(test_sine, (N_segment * i, N_samples - N_segment * i - N_segment)) for i in
            range(0, 20)]

    print(f"Taal Params: C1: {taal.C1}, C2: {taal.C2}, ratio: {taal.C1 / taal.C2}")
    print(f"Par Params: Ca: {par.Ca}, Cs: {par.Cs}, ratio: {par.Ca / par.Cs}")

    fig, axs = mpl.subplots(3, 3, figsize=(32,24), dpi=50)

    # Masking curve plot
    zero_signal = np.zeros(N_samples)
    axs[0,0].plot(zero_signal, 'black')
    axs[0,0].set_xlabel("samples")
    axs[0,0].set_ylabel("input signal")

    axs[0,1].semilogx(frequency, threshold_in_quiet_db(frequency), 'black')
    axs[0,1].semilogx(frequency, mapping.signal_to_pressure(taal.masking_threshold(zero_signal)), '--')
    axs[0,1].semilogx(frequency, mapping.signal_to_pressure( par.masking_threshold(zero_signal)), '-.')
    axs[0,1].legend(["Threshold in Quiet", "Taal Model", "Par Model"])
    axs[0,1].set_xlim((100, max(frequency)))
    axs[0,1].set_ylim((-20, 100))
    axs[0,1].set_xlabel("frequency [Hz]")
    axs[0,1].set_ylabel("SPL [dB Pa]")
    axs[0,1].set_title("Masking Curve")

    # Gain plot
    # for gain in taal.gain(zero_signal)[0:10:60]:
    #     axs[0,2].plot(gain, '--')
    # axs[0,2].legend(["Taal Model"])
    # axs[0,2].plot(np.mean(taal.gain(zero_signal)), '--')
    for idx, unit in enumerate(test_sine_signals):
        axs[0,2].scatter(N_segment * idx, taal.detectability_gain(zero_signal, unit), c=['b'])
        axs[0,2].scatter(N_segment * idx,  par.detectability_gain(zero_signal, unit), c=['r'])
        print(par.detectability_gain(zero_signal, unit))
    axs[0,2].set_ylim((0, 50))

    # Masking curve plot
    ramp = np.concatenate((np.linspace(0, 1, int(0.05 * N_samples)), np.ones(N_samples - 2 * int(0.05 * N_samples)), np.linspace(1, 0, int(0.05 * N_samples))))
    sine = mapping.pressure_to_signal(50) * ramp * np.cos(2 * np.pi * 1000 * np.arange(0, N_samples) / sampling_rate)
    sine_signal = window * np.pad(sine[0 : int(N_samples / 2)], (int(N_samples / 2), 0))
    axs[1,0].plot(sine_signal, 'black')
    axs[1,0].set_xlabel("samples")
    axs[1,0].set_ylabel("input signal")

    axs[1,1].semilogx(frequency, mapping.signal_to_pressure( np.sqrt(2) * np.fft.rfft(sine_signal) / N_samples), 'k--',
            alpha=0.3)
    axs[1,1].semilogx(frequency, threshold_in_quiet_db(frequency), 'black')
    axs[1,1].semilogx(frequency, mapping.signal_to_pressure(taal.masking_threshold(sine_signal)), '--')
    axs[1,1].semilogx(frequency, mapping.signal_to_pressure( par.masking_threshold(sine_signal)), '-.')
    axs[1,1].legend(["Input Spectrum", "Threshold in Quiet", "Taal Model", "Par Model"])
    axs[1,1].hlines(38 - 3, 0, sampling_rate / 2, 'black')
    axs[1,1].hlines(50 - 3, 0, sampling_rate / 2, 'black')
    axs[1,1].set_xlim((100, max(frequency)))
    axs[1,1].set_ylim((-20, 100))
    axs[1,1].set_xlabel("frequency [Hz]")
    axs[1,1].set_ylabel("SPL [dB Pa]")
    axs[1,1].set_title("Masking Curve")

    # Gain plot
    # for gain in mean(taal.gain(sine_signal)[0:10:60]:
    #     axs[1,2].plot(gain, '--')
    # axs[1,2].legend(["Taal Model"])
    for idx, unit in enumerate(test_sine_signals):
        axs[1,2].scatter(N_segment * idx, taal.detectability_gain(sine_signal, unit), c=['b'])
        axs[1,2].scatter(N_segment * idx,  par.detectability_gain(sine_signal, unit), c=['r'])
    axs[1,2].set_ylim((0, 50))

    # Masking curve plot
    sine_signal = window * sine
    axs[2,0].plot(sine_signal, 'black')
    axs[2,0].set_xlabel("samples")
    axs[2,0].set_ylabel("input signal")

    axs[2,1].semilogx(frequency, mapping.signal_to_pressure( np.sqrt(2) * np.fft.rfft(sine_signal) / N_samples), 'k--',
            alpha=0.3)
    axs[2,1].semilogx(frequency, threshold_in_quiet_db(frequency), 'black')
    axs[2,1].semilogx(frequency, mapping.signal_to_pressure(taal.masking_threshold(sine_signal)), '--')
    axs[2,1].semilogx(frequency, mapping.signal_to_pressure( par.masking_threshold(sine_signal)), '-.')
    axs[2,1].legend(["Input Spectrum", "Threshold in Quiet", "Taal Model", "Par Model"])
    axs[2,1].hlines(38 - 3, 0, sampling_rate / 2, 'black')
    axs[2,1].hlines(50 - 3, 0, sampling_rate / 2, 'black')
    axs[2,1].set_xlim((100, max(frequency)))
    axs[2,1].set_ylim((-20, 100))
    axs[2,1].set_xlabel("frequency [Hz]")
    axs[2,1].set_ylabel("SPL [dB Pa]")
    axs[2,1].set_title("Masking Curve")

    # Gain plot
    # for gain in taal.gain(sine_signal)[0:10:60]:
    #     axs[2,2].plot(gain, '--')
    # axs[2,2].legend(["Taal Model"])
    for idx, unit in enumerate(test_sine_signals):
        axs[2,2].scatter(N_segment * idx, taal.detectability_gain(sine_signal, unit), c=['b'])
        axs[2,2].scatter(N_segment * idx,  par.detectability_gain(sine_signal, unit), c=['r'])
    axs[2,2].set_ylim((0, 50))

    fig.savefig(os.path.join(os.path.dirname(__file__),"plots/test_masking_curve.png"))
