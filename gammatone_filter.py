import numpy as np
import scipy as sp

from scipy.special import factorial, factorial2
from perceptual_helpers import frequencies_to_erb

def gammatone_filter_frequency_response(frequencies : float, center_frequency : float, filter_order : int):
    erb_factor = gammatone_filter_erb_factor(filter_order)
    fd = frequencies - center_frequency
    fc = np.power(fd / (erb_factor * frequencies_to_erb(center_frequency)), 2)
    return np.power(1.0 + fc, -filter_order / 2.0)


def gammatone_filter_erb_factor(filter_order : int):
    sq = np.power(2, filter_order - 1)
    f1 = sp.special.factorial(filter_order - 1)
    f2 = sp.special.factorial2(2 * filter_order - 3)
    return (sq * f1) / (np.pi * f2)
