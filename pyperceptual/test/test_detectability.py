import pytest 
import os
import matplotlib.pyplot as mpl
import numpy as np
import scipy as sp 
import scipy.signal.windows

from pyperceptual.models.taal_model import taal_model
from pyperceptual.models.par_model import par_model
from pyperceptual.utility.signal_pressure_mapping import signal_pressure_mapping
from pyperceptual.utility.threshold_in_quiet import threshold_in_quiet

# Define problem parameters
sampling_rate = 48000.0
N_samples = int(0.002 * sampling_rate)
N_filters = 64
mapping = signal_pressure_mapping(1, 100)

@pytest.mark.parametrize("model", [
    taal_model(sampling_rate, N_samples, mapping, N_filters=N_filters),
     par_model(sampling_rate, N_samples, mapping, N_filters=N_filters)
])
def test_detectability_gain_correspondance(model):
    frequency = model.frequency_axis

    sine = lambda A, f : A * np.cos(2 * np.pi * f / sampling_rate * np.arange(0, N_samples))

    for freq in frequency:
        A_tq = threshold_in_quiet(freq, mapping)
        sine_tq = sine(A_tq, freq)
        x = np.zeros(N_samples)
        D_direct = model.detectability_direct(x, sine_tq)
        D_gain = model.detectability_gain(x, sine_tq)
