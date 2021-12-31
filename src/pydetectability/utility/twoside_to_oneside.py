import numpy as np
import math
def twoside_to_oneside_scaled(x: np.ndarray, axis: int = 1):
    # Remember if input is 1D
    one_dimensional_input = x.ndim == 1
    # If column-wise is desirable
    if axis == 0:
        x = x.T

   # Resize for processing
    x = np.array(x, ndmin=2)
    # Extract length
    N = x.shape[1]
    # Extract DC frequqency component
    x_dc = np.array(x[:, 0:1], ndmin=2)
    # Extract Nyquist frequqency component
    x_nq = np.array(x[:, math.ceil(N / 2 + 1) - 1 : math.floor(N / 2 + 1)], ndmin=2)
    # Extract other frequqency components
    x_fr = np.array(x[:, 1 : math.ceil(N / 2)], ndmin = 2)
    # Concatenate all components
    y = np.concatenate([
        x for x in [x_dc, 2 * x_fr, x_nq] if x.size > 0
    ], axis=1)

    # Resize back if needed
    if one_dimensional_input:
        y = y.reshape(-1)
    # Transpose back if needed
    if axis == 0:
        y = y.T

    # Return output value
    return y

def twoside_to_oneside(x: np.ndarray, axis: int = 1):
    # Remember if input is 1D
    one_dimensional_input = x.ndim == 1
    # If column-wise is desirable
    if axis == 0:
        x = x.T

    s = x.shape
    # Resize for processing
    x = np.array(x, ndmin=2)
    # Extract length
    N = x.shape[1]
    # Extract DC frequqency component
    x_dc = np.array(x[:, 0:1], ndmin=2)
    # Extract Nyquist frequqency component
    x_nq = np.array(x[:, math.ceil(N / 2 + 1) - 1 : math.floor(N / 2 + 1)], ndmin=2)
    # Extract other frequqency components
    x_fr = np.array(x[:, 1 : math.ceil(N / 2)], ndmin = 2)
    # Concatenate all components
    y = np.concatenate([
        x for x in [x_dc, x_fr, x_nq] if x.size > 0
    ], axis=1)
    # Reshape back if needed
    if len(s) == 1:
        y = y.reshape(-1)

    # Resize back if needed
    if one_dimensional_input:
        y = y.reshape(-1)
    # Transpose back if needed
    if axis == 0:
        y = y.T

    # Return output value
    return y
