import numpy as np


def power_spectrum(series):
    fft = np.fft.fft(series)
    ps = np.abs(fft) ** 2

    return ps
