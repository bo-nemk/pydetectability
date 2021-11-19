import pytest 
import os
import matplotlib.pyplot as mpl
import numpy as np

from pyperceptual.models.taal_model import taal_model
from pyperceptual.models.par_model import par_model
from pyperceptual.utility.signal_pressure_mapping import signal_pressure_mapping
from pyperceptual.utility.threshold_in_quiet import threshold_in_quiet_db

def test_masking_curve():
    # Define problem parameters
    N_samples = 1280
    N_filters = 64
    sampling_rate = 48000.0

    mapping = signal_pressure_mapping(1, 1)
    taal = taal_model(sampling_rate, N_samples, mapping, N_filters=N_filters)
    par = par_model(sampling_rate, N_samples, mapping, N_filters=N_filters)
    frequency = taal.frequency_axis

    print(f"Taal Params: C1: {taal.C1}, C2: {taal.C2}, ratio: {taal.C1 / taal.C2}")
    print(f"Par Params: Ca: {par.Ca}, Cs: {par.Cs}, ratio: {par.Ca / par.Cs}")

    fig, axs = mpl.subplots(2,2)
    axs[0,0].semilogx(frequency, threshold_in_quiet_db(frequency), 'black')
    axs[0,0].semilogx(frequency, mapping.signal_to_pressure(taal.masking_threshold(np.zeros(N_samples))), '--')
    axs[0,0].semilogx(frequency, mapping.signal_to_pressure( par.masking_threshold(np.zeros(N_samples))), '-.')
    axs[0,0].legend(["Threshold in Quiet", "Taal Model", "Par Model"])

    axs[0,1].plot(frequency, 1 / (taal.masking_threshold(np.zeros(N_samples)) ** 2), '--')
    axs[0,1].plot(frequency, 1 / (par.masking_threshold(np.zeros(N_samples)) ** 2), '-.')
    axs[0,1].legend(["Taal Model", "Par Model"])

    
    fig.savefig(os.path.join(os.path.dirname(__file__),"plots/test_masking_curve.png"))
