import numpy as np
import scipy as sp

from scipy.special import factorial, factorial2

class gammatone_filter_bank:
    def __init__(self, frequencies : float, center_frequencies : float, \
            filter_order : int = 4):
        # Map over input parameters
        self.frequencies = frequencies
        self.center_frequencies = center_frequency
        self.filter_order = filter_order

        # Pre-allocate bank
        self.filter_bank = np.zeros(center_frequencies.length, \
                frequencies.length)

        # Fill bank
        self.erb_factor = gammatone_filter_erb_factor()
        for i in range(1, center_frequencies.length + 1):
            self.filter_bank[i, :] = gammatone_filter(self.frequencies, \
                    self.center_frequency, self.erb_factor, self.filter_order)

    def gammatone_filter_erb_factor(self):
        sq = np.pow(2, self.filter_order - 1)
        f1 = sp.special.factorial(self.filter_order - 1)
        f2 = sp.special.factorial2(2 * self.filter_order - 3)
        return (sq * f1) / (np.pi * f2)

    def gammatone_filter(self, center_frequency : float):
        fd = self.frequencies - center_frequency
        fc = np.pow(fd / (erb_factor * frequencies_to_erb(center_frequency)), 2)
        return np.pow(1.0 + fc, -filter_order / 2.0)

    def frequencies_to_erb(self, frequencies : float):
        return 24.7 * (4.37 * (frequencies / 1000.0) + 1.0)
