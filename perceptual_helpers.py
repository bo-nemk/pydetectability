import numpy as np
import scipy as sp

def frequencies_to_erb(frequency: float):
    return 24.7 * (4.37 * (frequency / 1000.0) + 1.0)

def frequencies_to_erbs(frequency: float):
    return 21.4 * np.log10(1 + 0.00437 * frequency)

def erbs_to_frequencies(erbs: float):
    return (np.power(10, (erbs / 21.4)) - 1) / 0.00437

def erbspace(fmin: float, fmax: float, N_filters: int):
    return erbs_to_frequencies(np.linspace( \
        frequencies_to_erbs(fmin), \
        frequencies_to_erbs(fmax), N_filters))



