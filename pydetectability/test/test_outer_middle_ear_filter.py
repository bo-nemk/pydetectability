import pytest 
import matplotlib.pyplot as mpl
import numpy as np

from pydetectability.utility.outer_middle_ear_filter import outer_middle_ear_filter
from pydetectability.utility.signal_pressure_mapping import signal_pressure_mapping

# Create a test-plot of the low-pass filter
def test_outer_middle_ear_filter_plot():
    # Define problem parameters
    N_fft = 10240
    sampling_rate = 48000.0
    mapping = signal_pressure_mapping(1, 100)

    # Define x-axis
    frequencies = np.fft.rfftfreq(N_fft, 1 / sampling_rate)
    time = np.linspace(0, N_fft / sampling_rate, N_fft)

    # Create lowpass filter
    outer_middle_ear_filter_instance_freq = outer_middle_ear_filter(frequencies, mapping, N_fft).filter
    outer_middle_ear_filter_instance_time = np.fft.irfft(outer_middle_ear_filter_instance_freq, n=N_fft)

    # Create plots
    fig, axs = mpl.subplots(2)

    axs[0].semilogx(frequencies, 20 * np.log10(np.abs(outer_middle_ear_filter_instance_freq)))
    axs[0].set_title("Frequency Domain Outer-middle Ear Filter")
    axs[0].set_xlabel("frequency [Hz]")
    axs[0].set_ylabel("amplitude [dB]")
    axs[0].set_xlim(min(frequencies[1:]), max(frequencies)) 
    axs[0].set_ylim(-40, 120) 

    axs[1].plot(time, outer_middle_ear_filter_instance_time)
    axs[1].set_title("Time Domain Outer-middle Ear Filter")
    axs[1].set_xlabel("time [s]")
    axs[1].set_ylabel("amplitude")
    axs[1].set_xlim(0, N_fft / sampling_rate) 
    
    fig.tight_layout()
    fig.savefig("plots/test_outer_middle_ear_filter_plot.png")

    # If execution makes it this far, the test has passed.
    assert True


