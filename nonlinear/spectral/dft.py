import numpy as np


def dft(signal):
    """Compute the Discrete Fourier Transform (direct, O(N^2))."""
    N = len(signal)
    n = np.arange(N)
    k = n.reshape((N, 1))
    W = np.exp(-2j * np.pi * k * n / N)
    return W @ np.array(signal)


def idft(spectrum):
    """Compute the inverse DFT."""
    N = len(spectrum)
    n = np.arange(N)
    k = n.reshape((N, 1))
    W = np.exp(2j * np.pi * k * n / N)
    return (W @ np.array(spectrum)) / N


def frequency_axis(N, dt=1.0):
    """Return the frequency axis for a DFT of length N."""
    return np.fft.fftfreq(N, d=dt)
