import numpy as np
from nonlinear.chaos.ruelle_takens import classify_ruelle_takens, spectral_entropy
from nonlinear.chaos.melnikov import has_simple_zeros


def test_classify_periodic():
    # Pure sine wave should be classified as periodic
    t = np.linspace(0, 10 * np.pi, 1024)
    series = np.sin(t)
    result = classify_ruelle_takens(series)
    assert result == "periodic"


def test_classify_broadband():
    # White noise has much higher spectral entropy than periodic signal
    np.random.seed(42)
    noise = np.random.rand(1024)
    t = np.linspace(0, 10 * np.pi, 1024)
    sine = np.sin(t)
    assert spectral_entropy(noise) > spectral_entropy(sine) * 2


def test_spectral_entropy_sine_vs_noise():
    t = np.linspace(0, 10 * np.pi, 512)
    sine = np.sin(t)
    np.random.seed(42)
    noise = np.random.rand(512)

    h_sine = spectral_entropy(sine)
    h_noise = spectral_entropy(noise)
    # Noise has higher spectral entropy than a pure tone
    assert h_noise > h_sine


def test_has_simple_zeros_oscillating():
    M = [1, -1, 1, -1]
    has_zeros, count = has_simple_zeros(M)
    assert has_zeros is True
    assert count == 3


def test_has_simple_zeros_noncrossing():
    M = [1, 2, 3, 4]
    has_zeros, count = has_simple_zeros(M)
    assert has_zeros is False
    assert count == 0
