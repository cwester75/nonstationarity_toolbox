import numpy as np


def acf(series, lag):
    """Compute the autocorrelation function at a given lag.

    Returns 0.0 if the series has zero variance or if lag >= len(series).
    """
    s = np.array(series, dtype=float)

    if lag <= 0 or lag >= len(s):
        return 0.0

    mean = np.mean(s)
    den = np.sum((s - mean) ** 2)

    if den == 0:
        return 0.0

    num = np.sum((s[:-lag] - mean) * (s[lag:] - mean))
    return num / den
