import pytest 
import numpy as np

from pydetectability.utility.oneside_to_twoside import *
from pydetectability.utility.twoside_to_oneside import *

def test_rfft_forward_equivalence():
    N_fft = 1024
    x = np.random.randn(N_fft)

    assert (np.abs(twoside_to_oneside(np.fft.fft(x)) - np.fft.rfft(x)) < 1e-8).all()

def test_rfft_backward_equivalence():
    N_fft = 1024
    x = np.random.randn(N_fft)

    assert (np.abs(oneside_to_twoside(np.fft.rfft(x)) - np.fft.fft(x)) < 1e-8).all()
