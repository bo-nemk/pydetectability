import numpy as np
import scipy as sp

from signal_pressure_mapping import signal_pressure_mapping

def threshold_in_quiet_db(frequencies: float):
     return 3.64 * np.power(frequencies / 1000, -0.8) - 6.5 * \
        np.exp(-0.6 * np.power(frequencies / 1000 - 3.3, 2)) + \
        10e-4 * np.power(frequencies / 1000, 4)

def threshold_in_quiet(frequencies: float, mapping : signal_pressure_mapping):
     return mapping.pressure_to_signal(threshold_in_quiet_db(frequencies))
