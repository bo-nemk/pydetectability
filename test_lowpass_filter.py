import pytest 
import matplotlib.pyplot as mpl
import numpy as np

from lowpass_filter import lowpass_filter

# Create a test-plot of the low-pass filter
def test_lowpass_filter_plot():
    # Define problem parameters
    N_fft = 1024
    cutoff_frequency = 1000.0
    sampling_rate = 8000.0

    # Define x-axis
    frequencies = np.fft.rfftfreq(N_fft, 1 / sampling_rate)
    time = np.linspace(0, N_fft / sampling_rate, N_fft)

    # Create lowpass filter
    lowpass_filter_instance_freq = lowpass_filter(frequencies, cutoff_frequency, sampling_rate).filter
    lowpass_filter_instance_time = np.fft.irfft(lowpass_filter_instance_freq, n=N_fft)

    # Create plots
    fig, axs = mpl.subplots(2)

    axs[0].plot(frequencies, 20 * np.log10(np.abs(lowpass_filter_instance_freq)))
    axs[0].axhline(-3, 0, 1)
    axs[0].set_title("Frequency Domain Lowpass Filter")
    axs[0].set_xlabel("frequency [Hz]")
    axs[0].set_ylabel("amplitude [dB]")
    axs[0].set_xlim(min(frequencies[1:]), max(frequencies)) 

    axs[1].plot(time, lowpass_filter_instance_time)
    axs[1].set_title("Time Domain Lowpass Filter")
    axs[1].set_xlabel("time [s]")
    axs[1].set_ylabel("amplitude")
    axs[1].set_xlim(0, N_fft / sampling_rate) 

    fig.tight_layout()
    fig.savefig("plots/test_lowpass_filter_plot.png")

    # If execution makes it this far, the test has passed.
    assert True



