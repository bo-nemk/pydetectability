import numpy as np
import scipy as sp

class signal_pressure_mapping:
    def __init__(self, full_scale_amplitude: float, full_scale_amplitude_db: float):
        # Map over parameters
        self.full_scale_amplitude = full_scale_amplitude
        self.full_scale_amplitude_db = full_scale_amplitude_db

        # Compute the bias
        self.offset_db = self.full_scale_amplitude_db - 20 * np.log10(full_scale_amplitude)
   
    def signal_to_pressure(self, X: float):
        return 20 * np.log10(np.abs(X)) + self.offset_db

    def pressure_to_signal(self, X: float):
        return np.power(10, (X - self.offset_db) / 20)
