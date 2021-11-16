import numpy as np
import scipy as sp

from gammatone_filter import gammatone_filter_frequency_response
from perceptual_helpers import erbspace

class gammatone_filter_bank:
    def __init__(self, frequencies : float, sampling_rate: float, N_filters: int = 64, filter_order : int = 4):
        # Map over input parameters
        self.__frequencies = frequencies
        self.__sampling_rate = sampling_rate 
        self.__N_filters = N_filters
        self.__filter_order = filter_order

        # calculate center frequencies
        self.__center_frequencies = erbspace(0, self.__sampling_rate / 2, self.__N_filters) 
        print(self.__center_frequencies)

        # Pre-allocate bank
        self.filter_bank = np.zeros((self.__N_filters, self.__frequencies.size))

        # Fill bank
        for i in range(0, self.__N_filters):
            self.filter_bank[i, :] = gammatone_filter_frequency_response(self.__frequencies, \
                    self.__center_frequencies[i], self.__filter_order)
