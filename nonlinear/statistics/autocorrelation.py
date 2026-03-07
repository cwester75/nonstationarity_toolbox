import numpy as np


def acf(series, lag):
    s = np.array(series)
    mean = np.mean(s)

    num = np.sum((s[:-lag] - mean) * (s[lag:] - mean))
    den = np.sum((s - mean) ** 2)

    return num / den
