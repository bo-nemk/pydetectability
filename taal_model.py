import numpy as np 
import scipy as sp 

class taal_model:
    def __init__(self, sampling_rate : float, N_samples : int, \
            N_filters : int, filter_order : int = 4):
        # Map over input parameters
        self.sampling_rate = sampling_rate
        self.N_samples = N_samples
        self.N_filters = N_filters
        self.filter_order = filter_order

        # Compute frequency axis
        self.frequencies = np.linspace(0, N_samples / 2, N_samples) \
                / N_samples * sampling_rate

        # Compute min / max center frequencies
        self.min_frequency = 150.0
        self.max_frequency = min(8000.0, sampling_rate / 2)

        # Convert to min / max center erbs
        self.min_erbs = self.frequencies_to_erbs(self.min_frequency)
        self.max_erbs = self.frequencies_to_erbs(self.max_frequency)

        # Compute center frequencies
        self.center_erbs = np.linspace(self.min_erbs, self.max_erbs, N_filters)
        self.center_frequencies = self.erbs_to_frequencies(self.center_erbs)

    def frequencies_to_erbs(self, frequencies):
        return 21.4 * np.log10(4.37 * (frequencies / 1000.0) + 1.0)

    def erbs_to_frequencies(self, erbs):
        return (np.pow(10.0, erbs / 21.4) - 1.0) / 4.37 * 1000.0
