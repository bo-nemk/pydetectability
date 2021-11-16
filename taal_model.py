import numpy as np 
import scipy as sp 
import scipy.signal
import scipy.optimize

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
        self.auditory_filter_bank_freq = auditory_filter_bank(self.frequency_axis, 
                self.sampling_rate, self.mapping).filter_bank
        self.auditory_filter_bank_time = np.fft.irfft(self.auditory_filter_bank_freq)
        self.N_auditory_filter = self.auditory_filter_bank_time[0].size
        
        # Compute lowpass filter
        self.N_convolved = self.N_samples + self.N_auditory_filter - 1
        self.convolved_frequency_axis = np.fft.rfftfreq(self.N_convolved, 1 / self.sampling_rate);
        self.lowpass_freq = lowpass_filter(self.convolved_frequency_axis, 1000, self.sampling_rate).filter_freq
        self.lowpass_time = np.fft.irfft(self.lowpass_freq)

        # Coefficients
        training_rate = 10000
        sine_base = np.cos(2 * np.pi * training_rate / sampling_rate * np.arange(0, self.N_samples))
        sine_base_internal = self.__internal_representation(sine_base)
        sine_threshold_in_quiet = threshold_in_quiet(training_rate, self.mapping) * sine_base
        sine_threshold_in_quiet_internal = self.__internal_representation(sine_threshold_in_quiet)
        sine_masker = self.mapping.pressure_to_signal(70) * sine_base
        sine_masker_internal = self.__internal_representation(sine_masker)
        sine_masked = self.mapping.pressure_to_signal(52) * sine_base
        sine_masked_internal = self.__internal_representation(sine_masked)

        
        helper_function = lambda x, C2 : C2 * sum(np.linalg.norm(x, 1.0, axis=1))
        bisection_function = lambda C2 : helper_function(sine_masked_internal / (sine_masker_internal +
            helper_function(sine_threshold_in_quiet_internal, C2)), C2) - 1

        self.C2 = sp.optimize.bisect(bisection_function, 0, 100000)
        print(self.C2)
        self.C1 = helper_function(sine_threshold_in_quiet_internal, self.C2)
        print(self.C1)

    def gain_function(self, x):
        g = np.zeros((self.N_filters, self.N_convolved))
        y = self.__internal_representation(x)
        for i in range(0, self.N_filters):
            # Determine energy ratio
            fr   = self.C2 / (y[i] + self.C1)
            # Return filtered energy ratio
            g[i] = np.sqrt(np.fft.irfft(np.fft.rfft(fr) * self.lowpass_freq, n=self.N_convolved))

        return g

    def __internal_representation(self, x):
        y = np.zeros((self.N_filters, self.N_convolved))
        for i in range(0, self.N_filters):
            xi   = np.convolve(x, self.auditory_filter_bank_time[i])
            y[i] = np.fft.irfft(np.fft.rfft(np.power(np.abs(xi), 2.0)) * self.lowpass_freq, n=self.N_convolved)

        return y;

    def detectability(self, x, e):
        D = 0
        g = self.gain_function(x)
        for i in range(0, self.N_filters):
            # Determine auditory-filtered e
            ei = np.convolve(e, self.auditory_filter_bank_time[i])
            D  = D + (np.linalg.norm(g[i] * ei) ** 2)

        return D


    def masking_threshold(self, x):
        t = np.zeros(self.convolved_frequency_axis.size)
        for i in range(0, self.convolved_frequency_axis.size):
            f = self.convolved_frequency_axis[i]
            e = np.cos(2 * np.pi * f * np.arange(0, self.N_samples) / self.sampling_rate)
            t[i] = 1 / np.sqrt(self.detectability(x, e))

        return t
