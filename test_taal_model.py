import pytest 
import matplotlib.pyplot as mpl
import numpy as np

from taal_model import taal_model
from signal_pressure_mapping import signal_pressure_mapping
from threshold_in_quiet import threshold_in_quiet_db

def test_taal_model():
    # Define problem parameters
    N_samples = 12800
    sampling_rate = 48000.0
    model = taal_model(sampling_rate, N_samples, N_filters=64)
    mapping = signal_pressure_mapping(1, 100)

    # Create masking threshold
    x = np.zeros(N_samples)
    threshold_freq = model.masking_threshold(x)
    gain = model.gain_function(x)

    # Test
    fig, axs = mpl.subplots(2)

    axs[0].semilogx(model.frequency_axis, mapping.signal_to_pressure(threshold_freq))
    axs[0].semilogx(model.frequency_axis, threshold_in_quiet_db(model.frequency_axis))
    axs[0].set_title("Taal Model Masking Threshold")
    axs[0].set_xlabel("frequency [Hz]")
    axs[0].set_ylabel("amplitude [dB]")
    axs[0].set_xlim(min(model.frequency_axis), max(model.frequency_axis)) 
    axs[0].set_ylim(-20, 150) 

    axs[1].plot(gain[0,:])
    axs[1].set_title("Taal Model Gain")
    axs[1].set_xlabel("frequency [Hz]")
    axs[1].set_ylabel("amplitude [dB]")
    # axs[1].set_xlim(min(model.convolved_frequency_axis), max(model.convolved_frequency_axis)) 
    # axs[1].set_ylim(-20, 150) 

    fig.tight_layout()
    fig.savefig("plots/test_taal_model.png")

