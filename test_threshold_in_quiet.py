import pytest 
import matplotlib.pyplot as mpl
import numpy as np

from threshold_in_quiet import threshold_in_quiet_db, threshold_in_quiet
from signal_pressure_mapping import signal_pressure_mapping

def test_outer_threshold_in_quiet():
    # Define problem parameters
    N_fft = 1024
    sampling_rate = 48000.0
    mapping = signal_pressure_mapping(1, 100)

    # Define x-axis
    frequencies = np.fft.rfftfreq(N_fft, 1 / sampling_rate)
    time = np.linspace(0, N_fft / sampling_rate, N_fft)

    # Create threshold
    threshold_freq = threshold_in_quiet_db(frequencies)
    threshold_time = np.fft.irfft(threshold_in_quiet(frequencies, mapping), n=N_fft)

    # Create plots
    fig, axs = mpl.subplots(2)

    axs[0].semilogx(frequencies, threshold_freq)
    axs[0].set_title("Frequency Domain Threshold in Quiet")
    axs[0].set_xlabel("frequency [Hz]")
    axs[0].set_ylabel("amplitude [dB]")
    axs[0].set_xlim(min(frequencies[1:]), max(frequencies)) 
    axs[0].set_ylim(-40, 180) 

    axs[1].plot(time, threshold_time)
    axs[1].set_title("Time Domain Threshold in Quiet")
    axs[1].set_xlabel("time [s]")
    axs[1].set_ylabel("amplitude")
    axs[1].set_xlim(0, N_fft / sampling_rate) 
    
    fig.tight_layout()
    fig.savefig("plots/test_threshold_in_quiet.png")

    # If execution makes it this far, the test has passed.
    assert True


