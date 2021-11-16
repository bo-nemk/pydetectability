import pytest 
import matplotlib.pyplot as mpl
import numpy as np

from gammatone_filter_bank import gammatone_filter_bank

# Create a test-plot of the low-pass filter
def test_gammatone_filter_bank_plot():
    # Define problem parameters
    N_fft = 1024
    N_filters = 10
    sampling_rate = 48000.0

    # Define x-axis
    frequencies = np.fft.rfftfreq(N_fft, 1 / sampling_rate)
    time = np.linspace(0, N_fft / sampling_rate, N_fft)

    # Create filter bank
    filter_bank = gammatone_filter_bank(frequencies, sampling_rate, N_fft, N_filters=N_filters)
    gammatone_filter_bank_freq = filter_bank.filter_bank_freq
    gammatone_filter_bank_time = filter_bank.filter_bank_time

    # Create plots
    fig, axs = mpl.subplots(2)

    for gammatone_filter_freq in gammatone_filter_bank_freq:
        axs[0].plot(frequencies, 20 * np.log10(np.abs(gammatone_filter_freq)))

    axs[0].set_title("Frequency Domain Gammatone Filters")
    axs[0].set_xlabel("frequency [Hz]")
    axs[0].set_ylabel("amplitude [dB]")

    for gammatone_filter_time in gammatone_filter_bank_time:
        axs[1].plot(time, gammatone_filter_time)
    axs[1].set_title("Time Domain Gammatone Filter Bank")
    axs[1].xlabel = "frequency [Hz]"
    axs[1].ylabel = "amplitude"

    fig.tight_layout()
    fig.savefig("plots/test_gammatone_filter_bank_plot.png")

    # If execution makes it this far, the test has passed.
    assert True



