import numpy as np
from nonlinear.spectral.fft_tools import power_spectrum
from nonlinear.spectral.dft import dft, idft


def test_power_spectrum_length():
    signal = np.random.rand(128)
    ps = power_spectrum(signal)
    assert len(ps) == 128


def test_power_spectrum_nonnegative():
    signal = np.random.rand(64)
    ps = power_spectrum(signal)
    assert np.all(ps >= 0)


def test_dft_matches_numpy():
    signal = np.random.rand(32)
    our_dft = dft(signal)
    np_fft = np.fft.fft(signal)
    assert np.allclose(our_dft, np_fft, atol=1e-10)


def test_idft_roundtrip():
    signal = np.random.rand(32)
    reconstructed = idft(dft(signal))
    assert np.allclose(signal, reconstructed.real, atol=1e-10)
