import numpy as np 
import scipy as sp 
import scipy.signal
import scipy.optimize

import matplotlib.pyplot as plt

from pydetectability.utility.auditory_filter_bank import auditory_filter_bank
from pydetectability.utility.signal_pressure_mapping import signal_pressure_mapping
from pydetectability.utility.lowpass_filter import lowpass_filter
from pydetectability.utility.threshold_in_quiet import threshold_in_quiet

np.seterr(divide='ignore')

class taal_model:
    def __apply_auditory_filter_bank(self, s):
        assert(s.size == self.__N_samples)
        return [np.fft.irfft(np.fft.rfft(s) * self.auditory_filter_bank_freq[i], n=self.__N_samples) for i in range(0, self.__N_filters)]

    def __apply_lowpass_filter(self, s):
        return np.fft.irfft(np.fft.rfft(s) * self.lowpass_filter_freq, n=self.__N_samples)

    def __internal_representation(self, s):
        return self.__apply_lowpass_filter(np.power(np.abs(self.__apply_auditory_filter_bank(s)), 2.0))
            
    def __detectability(self, x, e, C1, C2):
        return C2 * np.abs(self.__internal_representation(e) / (self.__internal_representation(x) + C1)).sum(axis=1).sum()

    def __sine(self, amplitude, rate):
        return amplitude * np.cos(2 * np.pi * rate / self.__sampling_rate * np.arange(0, self.__N_samples))


    def __init__(self,sampling_rate : float, N_samples : int, mapping, N_filters : int = 64, filter_order : int = 4,
            training_rate : int = 1000):
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
        self.auditory_filter_bank_freq = auditory_filter_bank_object.filter_bank_freq
        
        # Compute lowpass filter
        lowpass_filter_object = lowpass_filter(self.frequency_axis, 1000, self.__sampling_rate)
        self.lowpass_filter_freq = lowpass_filter_object.filter_freq 

        # Training
        sine_zero = self.__sine(0, training_rate)
        sine_threshold_in_quiet = self.__sine(threshold_in_quiet(training_rate, self.__mapping), training_rate) 
        sine_masker = self.__sine(self.__mapping.pressure_to_signal(70), training_rate) 
        sine_masked = self.__sine(self.__mapping.pressure_to_signal(52), training_rate)
        
        # self.masking_helper = []
        self.C2 = sp.optimize.bisect(lambda C2 : self.__detectability(sine_masker, sine_masked,
            self.__detectability(sine_zero, sine_threshold_in_quiet, 1, C2), C2) - 1, 0, 1000)
        self.C1 = self.__detectability(sine_zero, sine_threshold_in_quiet, 1, self.C2)

    def detectability_direct(self, x, e):
        assert(e.size == self.__N_samples)
        assert(x.size == self.__N_samples)
        return self.__detectability(x, e, self.C1, self.C2)

    def gain(self, x):
        assert(x.size == self.__N_samples)
        return np.sqrt(self.__apply_lowpass_filter(self.C2 / (self.__internal_representation(x) + self.C1)))

    def detectability_gain(self, x, e):
        assert(e.size == self.__N_samples)
        assert(x.size == self.__N_samples)
        gi = self.gain(x)
        ei = self.__apply_auditory_filter_bank(e)
        return np.power(np.linalg.norm(gi * ei), 2.0).sum()

    def masking_threshold_brute_force(self, x):
        assert(x.size == self.__N_samples)
        test_sinusoids = [np.cos(2 * np.pi * self.frequency_axis[i] / self.__sampling_rate * np.arange(0,self.__N_samples)) for i in range(0, self.frequency_axis.size)]
        D = [self.detectability_direct(x, e) for e in test_sinusoids]
        return 1 / np.sqrt(D)

    def masking_threshold(self, x):
        # Through the approximation in taal2012
        assert(x.size == self.__N_samples)
        window = sp.signal.windows.hann(self.__N_samples)
        gi = self.gain(x)

        t = np.zeros(self.frequency_axis.size)
        for i in range(1, self.__N_filters):
            p = np.fft.fft(np.power(window * gi[i], 2.0))
            t = t + np.power(self.auditory_filter_bank_freq[i], 2.0) * (0.5 * p[0] + np.real(p[0:2:self.__N_samples]))

        return np.sqrt(1 / t)
