import pytest 
import matplotlib.pyplot as mpl
import numpy as np

from taal_model import taal_model
from signal_pressure_mapping import signal_pressure_mapping
from threshold_in_quiet import threshold_in_quiet_db

def test_taal_model():
    # Define problem parameters
    N_samples = 512
    sampling_rate = 96000.0
    model = taal_model(sampling_rate, N_samples)
    mapping = signal_pressure_mapping(1, 100)

    # Create masking threshold
    x = np.zeros(N_samples)
    threshold_freq = model.masking_threshold(x)

    # Test
    fig, axs = mpl.subplots(2)

    axs[0].semilogx(model.convolved_frequency_axis, mapping.signal_to_pressure(threshold_freq))
    axs[0].semilogx(model.convolved_frequency_axis, threshold_in_quiet_db(model.convolved_frequency_axis))
    axs[0].set_title("Taal Model Masking Threshold")
    axs[0].set_xlabel("frequency [Hz]")
    axs[0].set_ylabel("amplitude [dB]")
    print(model.convolved_frequency_axis)
    axs[0].set_xlim(min(model.convolved_frequency_axis[1:]), max(model.convolved_frequency_axis)) 
    axs[0].set_ylim(-20, 150) 

    fig.tight_layout()
    fig.savefig("plots/test_taal_model.png")

