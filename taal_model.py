import numpy as np 
import scipy as sp 
import scipy.signal
import scipy.optimize

import matplotlib.pyplot as plt

from auditory_filter_bank import auditory_filter_bank
from signal_pressure_mapping import signal_pressure_mapping
from lowpass_filter import lowpass_filter
from threshold_in_quiet import threshold_in_quiet

class taal_model:
    def __init__(self, sampling_rate : float, N_samples : int, \
            N_filters : int = 64, filter_order : int = 4):
        # Map over input parameters
        self.sampling_rate = sampling_rate
        self.N_samples = N_samples
        self.N_filters = N_filters
        self.filter_order = filter_order

        # Compute frequency axis
        self.frequency_axis = np.fft.rfftfreq(self.N_samples, 1 / self.sampling_rate);

        # Compute signal <-> db SPL
        self.mapping = signal_pressure_mapping(1, 100)

        # Compute ear filter bank
        self.auditory_filter_bank_freq = auditory_filter_bank(
                self.frequency_axis, 
                self.sampling_rate, 
                self.mapping, 
                N_filters=self.N_filters, 
                filter_order=self.filter_order).filter_bank

        self.auditory_filter_bank_time = np.fft.irfft(self.auditory_filter_bank_freq, n=N_samples, axis=1)
        
        # Compute lowpass filter
        self.N_convolved = 2 * self.N_samples - 1
        self.convolved_frequency_axis = np.fft.rfftfreq(self.N_convolved, 1 / self.sampling_rate);
        self.lowpass_freq = lowpass_filter(self.convolved_frequency_axis, 1000, self.sampling_rate).filter_freq

        # Coefficients
        training_rate = 1000
        sine_base = np.cos(2 * np.pi * training_rate / sampling_rate * np.arange(0, self.N_samples))
        sine_base_internal = self.__internal_representation(sine_base)
        sine_threshold_in_quiet = threshold_in_quiet(training_rate, self.mapping) * sine_base
        sine_threshold_in_quiet_internal = self.__internal_representation(sine_threshold_in_quiet)
        sine_masker = self.mapping.pressure_to_signal(70) * sine_base
        sine_masker_internal = self.__internal_representation(sine_masker)
        sine_masked = self.mapping.pressure_to_signal(69) * sine_base
        sine_masked_internal = self.__internal_representation(sine_masked)
        
        helper_function = lambda x, C2 : C2 * sum(np.linalg.norm(x, 1.0, axis=1))
        bisection_function = lambda C2 : helper_function(sine_masked_internal / (sine_masker_internal +
            helper_function(sine_threshold_in_quiet_internal, C2)), C2) - 1

        self.C2 = sp.optimize.bisect(bisection_function, 0, 100000)
        self.C1 = helper_function(sine_threshold_in_quiet_internal, self.C2)

    def gain_function(self, x):
        g = np.zeros((self.N_filters, self.N_convolved))
        y = self.__internal_representation(x)
        for i in range(0, self.N_filters):
            # Determine energy ratio
            fr = (self.C2 * np.ones(self.N_convolved)) / (y[i] + self.C1 * np.ones(self.N_convolved))
            # Return filtered energy ratio
            g[i] = np.sqrt(np.fft.irfft(np.fft.rfft(fr) * self.lowpass_freq, n=self.N_convolved))

        return g

    def __internal_representation(self, x):
        y = np.zeros((self.N_filters, self.N_convolved))
        for i in range(0, self.N_filters):
            xi   = np.power(np.abs(np.convolve(x, self.auditory_filter_bank_time[i])), 2.0)
            y[i] = np.fft.irfft(np.fft.rfft(xi) * self.lowpass_freq, n=self.N_convolved)

        return y;

    def detectability(self, x, e):
        D = self.C2 * sum(np.linalg.norm((self.__internal_representation(e)) / (self.__internal_representation(x) + self.C1), 1.0, axis=1))
        print(D)
        return D

    def masking_threshold(self, x):
        gi = self.gain_function(x)
        w  = np.power(sp.signal.windows.hann(self.N_convolved), 2.0)
        t  = np.zeros(self.frequency_axis.size)
        for k in range(0, self.frequency_axis.size):
            s  = 0
            for i in range(0, self.N_filters):
                gih = np.fft.fft(w * gi[i])
                hh = np.fft.fft(self.auditory_filter_bank_time[i])
                s = s + np.power(hh[k], 2.0) * (0.5 * gih[0] + np.real(gih[2 * k]))
            t[k] = 1 / np.sqrt(s)

        return t
