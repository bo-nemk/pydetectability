import numpy as np
import scipy as sp

from gammatone_filter import gammatone_filter_frequency_response
from perceptual_helpers import erbspace

class gammatone_filter_bank:
    def __init__(self, frequencies : float, sampling_rate: float, N_samples: int, N_filters: int = 64, filter_order : int = 4):
        # Map over input parameters
        self.__frequencies = frequencies
        self.__sampling_rate = sampling_rate 
        self.__N_filters = N_filters
        self.__filter_order = filter_order
        self.__N_time = N_samples
        self.__N_freq = self.__frequencies.size

        # calculate center frequencies
        self.__center_frequencies = erbspace(0, self.__sampling_rate / 2, self.__N_filters) 

        # Pre-allocate bank
        self.filter_bank_freq = np.zeros((self.__N_filters, self.__N_freq))
        self.filter_bank_time = np.zeros((self.__N_filters, self.__N_time))

        # Fill bank
        for i in range(0, self.__N_filters):
            gammatone = gammatone_filter_frequency_response(self.__frequencies, self.__center_frequencies[i],
                    self.__filter_order).astype(complex)
            delay = np.exp(-1j * 2 * np.pi * (self.__N_time / 2) * np.arange(0, self.__N_freq) / self.__N_time)
            self.filter_bank_freq[i] = gammatone * delay
            print(self.filter_bank_freq[i])
            self.filter_bank_time[i] = np.fft.irfft(self.filter_bank_freq[i], n=self.__N_time)

