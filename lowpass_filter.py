import numpy as np
import scipy as sp

class lowpass_filter:
    def __init__(self, frequencies: float, cutoff_frequency: float, \
            sampling_rate : float):
        # Map over input parameters
        self.__frequencies = frequencies
        self.__cutoff_frequency = cutoff_frequency 
        self.__sampling_rate = sampling_rate

        # Create filter
        self.filter = self.frequency_response()

    def a_factor(self):
        return -1 * np.exp(-2 * np.pi * self.__cutoff_frequency / self.__sampling_rate)

    def frequency_response(self):
        a = self.a_factor()
        return (1 + a) / np.sqrt(1 + a ** 2 + 2 * a * \
                np.cos(2 * np.pi * self.__frequencies / self.__sampling_rate))


