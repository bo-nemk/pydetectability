import numpy as np
import scipy as sp

from signal_pressure_mapping import signal_pressure_mapping
from threshold_in_quiet import threshold_in_quiet

class outer_middle_ear_filter:
    def __init__(self, frequencies: float, mapping : signal_pressure_mapping):
        self.__frequencies = frequencies
        self.__mapping = mapping

        threshold = threshold_in_quiet(self.__frequencies, self.__mapping)
        self.filter = 1 / threshold
