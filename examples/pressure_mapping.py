import numpy as np

reference_spl = 93.98

rms = lambda x : np.sqrt(np.mean(np.power(x, 2.0)))
signal_to_spl_mapping = lambda x, target_spl : (x / rms(x)) * np.power(10, ((target_spl - reference_spl) / 20))
pressure = lambda x : 20 * np.log10(rms(x)) + reference_spl

fs = 4800
test_signal = np.cos(2 * np.pi * 1000 * np.arange(0, 1000) / fs)

print(f"Expected RMS of test signal: {1 / np.sqrt(2)}")
print(f"Computed RMS of test signal: {rms(test_signal)}")
print(f"SPL of test signal: {pressure(signal_to_spl_mapping(test_signal, 50.0))}")
