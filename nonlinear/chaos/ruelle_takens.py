import numpy as np
from nonlinear.spectral.fft_tools import power_spectrum


def count_dominant_frequencies(series, threshold_ratio=0.1):
    """Count the number of dominant frequencies in a power spectrum.

    Used to detect transitions: periodic (1) -> quasi-periodic (2-3) -> chaotic (broad).
    """
    ps = power_spectrum(series)
    N = len(ps)
    ps_half = ps[:N // 2]
    max_power = np.max(ps_half)

    dominant = np.sum(ps_half > threshold_ratio * max_power)
    return dominant


def classify_ruelle_takens(series, threshold_ratio=0.1):
    """Classify dynamics according to Ruelle-Takens-Newhouse scenario.

    Returns one of: 'periodic', 'quasi-periodic', 'chaotic'.
    """
    n_freq = count_dominant_frequencies(series, threshold_ratio)

    if n_freq <= 2:
        return "periodic"
    elif n_freq <= 10:
        return "quasi-periodic"
    else:
        return "chaotic"


def spectral_entropy(series):
    """Compute spectral entropy as a measure of spectral complexity."""
    ps = power_spectrum(series)
    N = len(ps)
    ps_half = ps[:N // 2]

    total = np.sum(ps_half)
    if total == 0:
        return 0.0

    p = ps_half / total
    p = p[p > 0]
    return -np.sum(p * np.log2(p))
