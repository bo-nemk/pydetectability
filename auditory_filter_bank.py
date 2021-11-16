import numpy as np
import scipy as sp

from signal_pressure_mapping import signal_pressure_mapping
from gammatone_filter_bank import gammatone_filter_bank
from outer_middle_ear_filter import outer_middle_ear_filter

class auditory_filter_bank:
    def __init__(self, frequencies: float, sampling_rate: float, mapping: signal_pressure_mapping, \
            N_filters: int = 64, filter_order: int = 4):
        # Map over input parameters
        self.__frequencies = frequencies
        self.__sampling_rate = sampling_rate 
        self.__mapping = mapping
        self.__N_filters = N_filters
        self.__filter_order = filter_order

        # Create filters
        self.__gammatone_filter_bank = gammatone_filter_bank(self.__frequencies, self.__sampling_rate, N_filters=self.__N_filters)
        self.__outer_middle_ear_filter = outer_middle_ear_filter(self.__frequencies, self.__mapping)

        # Pre-allocate bank
        self.filter_bank = np.zeros((self.__N_filters, self.__frequencies.size))

        # Fill bank
        for i in range(0, self.__N_filters):
            self.filter_bank[i, :] = self.__outer_middle_ear_filter.filter * self.__gammatone_filter_bank.filter_bank[i]
