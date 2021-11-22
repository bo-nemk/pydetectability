import numpy as np 
import scipy as sp 
import scipy.signal
import scipy.optimize

import matplotlib.pyplot as plt

from pyperceptual.utility.auditory_filter_bank import auditory_filter_bank
from pyperceptual.utility.signal_pressure_mapping import signal_pressure_mapping
from pyperceptual.utility.threshold_in_quiet import threshold_in_quiet

np.seterr(divide='ignore')

class par_model:
    def __apply_auditory_filter_bank(self, s):
        assert(s.size == self.frequency_axis.size)
        return [s * self.__auditory_filter_bank_freq[i] for i in range(0, self.__N_filters)]

    def __internal_representation(self, s):
        assert(s.size == self.frequency_axis.size)
        return np.power(np.abs(self.__apply_auditory_filter_bank(s)), 2.0)
            
    def __detectability(self, x, e, Ca, Cs):
        assert(x.size == self.frequency_axis.size)
        assert(e.size == self.frequency_axis.size)
        return Cs * (self.__internal_representation(e).sum(axis=1) / (self.__internal_representation(x).sum(axis=1) + Ca)).sum()

    def __sine(self, amplitude, rate):
        return amplitude * np.cos(2 * np.pi * rate / self.__sampling_rate * np.arange(0, self.__N_samples))


    def __init__(self,sampling_rate : float, N_samples : int, mapping, N_filters : int = 64, filter_order : int = 4):
        # Map over input parameters
        self.__sampling_rate = sampling_rate
        self.__filter_order = filter_order
        self.__N_samples = N_samples
        self.__N_filters = N_filters
        self.__mapping = mapping

        # Compute frequency axis
        self.frequency_axis = np.fft.rfftfreq(self.__N_samples, d=(1 / self.__sampling_rate))

        # Compute ear filter bank
        auditory_filter_bank_object = auditory_filter_bank(self.frequency_axis, self.__sampling_rate, self.__mapping, 
                self.__N_samples, N_filters=self.__N_filters, filter_order=self.__filter_order)
        self.__auditory_filter_bank_freq = auditory_filter_bank_object.filter_bank_freq
        
        # Training
        training_rate = 1000
        sine_zero = self.__sine(0, training_rate)
        sine_threshold_in_quiet = self.__sine(threshold_in_quiet(training_rate, self.__mapping), training_rate) 
        sine_masker = self.__sine(self.__mapping.pressure_to_signal(70), training_rate) 
        sine_masked = self.__sine(self.__mapping.pressure_to_signal(52), training_rate)
        
        # self.masking_helper = []
        self.Cs = sp.optimize.bisect(lambda Cs : self.__detectability(np.fft.rfft(sine_masker), np.fft.rfft(sine_masked),
            self.__detectability(np.fft.rfft(sine_zero), np.fft.rfft(sine_threshold_in_quiet), 1, Cs), Cs) - 1, 0, 1000)
        self.Ca = self.__detectability(np.fft.rfft(sine_zero), np.fft.rfft(sine_threshold_in_quiet), 1, self.Cs)

    def detectability_direct(self, x, e):
        assert(e.size == self.__N_samples)
        assert(x.size == self.__N_samples)
        return self.__detectability(np.fft.rfft(x), np.fft.rfft(e), self.Ca, self.Cs)

    def gain(self, x):
        assert(x.size == self.__N_samples)
        xh = np.fft.rfft(x)
        return np.sqrt((np.power(self.__auditory_filter_bank_freq, 2.0) * self.Cs) / \
            (np.tile(self.__internal_representation(xh).sum(axis=1), (self.frequency_axis.size, 1)).T + self.Ca))
 
    def detectability_gain(self, x, e):
        assert(e.size == self.__N_samples)
        assert(x.size == self.__N_samples)
        gh = self.gain(x)
        eh = np.fft.rfft(e)
        return np.power(np.linalg.norm(gh * eh), 2.0) 

    def masking_threshold(self, x):
        assert(x.size == self.__N_samples)
        test_sinusoids = [np.cos(2 * np.pi * self.frequency_axis[i] / self.__sampling_rate * np.arange(0,self.__N_samples)) for i in range(0, self.frequency_axis.size)]
        D = [self.detectability_direct(x, e) for e in test_sinusoids]
        return 1 / np.sqrt(D)
