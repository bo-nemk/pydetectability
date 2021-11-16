import pytest 
import matplotlib.pyplot as mpl
import numpy as np

from auditory_filter_bank import auditory_filter_bank
from signal_pressure_mapping import signal_pressure_mapping

# Create a test-plot of the auditory filter bank
def test_auditory_filter_bank_plot():
    # Define problem parameters
    N_fft = 1024
    N_filters = 64
    sampling_rate = 96000.0
    mapping = signal_pressure_mapping(1, 100)

    # Define x-axis
    frequencies = np.fft.rfftfreq(N_fft, 1 / sampling_rate)
    time = np.linspace(0, N_fft / sampling_rate, N_fft)

    # Create filter bank
    auditory_filter_bank_freq = auditory_filter_bank(frequencies, sampling_rate, mapping, N_filters=N_filters).filter_bank
    auditory_filter_bank_time = np.fft.irfft(auditory_filter_bank_freq)

    # Create plots
    fig, axs = mpl.subplots(2)

    for auditory_filter_freq in auditory_filter_bank_freq:
        axs[0].plot(frequencies, 20 * np.log10(np.abs(auditory_filter_freq)))

    axs[0].set_title("Frequency Domain Auditory Filters")
    axs[0].set_xlabel("frequency [Hz]")
    axs[0].set_ylabel("amplitude [dB]")
    axs[0].set_xlim(min(frequencies[1:]), max(frequencies)) 
    axs[0].set_ylim(-50, 150) 

    for auditory_filter_time in auditory_filter_bank_time:
        axs[1].plot(time, auditory_filter_time)
    axs[1].set_title("Time Domain Auditory Filter Bank")
    axs[1].xlabel = "frequency [Hz]"
    axs[1].ylabel = "amplitude"
    axs[1].set_xlim(0, N_fft / sampling_rate) 

    fig.tight_layout()
    fig.savefig("plots/test_auditory_filter_bank_plot.png")

    # If execution makes it this far, the test has passed.
    assert True



