"""
Phase-space reconstruction via Takens delay embedding.

Links a scalar time series to a dynamical-systems representation
in reconstructed state space.
"""

import numpy as np


class PhaseSpace:
    """Takens delay-embedding of a scalar time series.

    Parameters
    ----------
    series : :class:`~nonlinear.core.timeseries.TimeSeries` or array-like
        The input time series.
    """

    def __init__(self, series):
        from .timeseries import TimeSeries
        if isinstance(series, TimeSeries):
            self._data = series.data
        else:
            self._data = np.asarray(series, dtype=float)

    def embed(self, dimension, delay=1):
        """Construct delay-coordinate vectors.

        Parameters
        ----------
        dimension : int
            Embedding dimension *m*.
        delay : int
            Time delay *tau* (default 1).

        Returns
        -------
        np.ndarray
            Array of shape ``(M, dimension)`` where
            ``M = N - (dimension - 1) * delay``.
        """
        x = self._data
        N = len(x)
        M = N - (dimension - 1) * delay
        if M < 1:
            raise ValueError(
                f"Series length {N} too short for dimension={dimension}, "
                f"delay={delay} (need at least {(dimension - 1) * delay + 1})"
            )
        return np.array([x[i: i + dimension * delay: delay] for i in range(M)])

    def optimal_delay(self, max_lag=50):
        """Estimate optimal delay via first minimum of autocorrelation.

        Scans autocorrelation from lag 1 to *max_lag* and returns the lag
        at the first local minimum (or *max_lag* if none found).
        """
        x = self._data
        mu = np.mean(x)
        var = np.sum((x - mu) ** 2)
        if var == 0:
            return 1

        prev = 1.0  # acf at lag 0
        for lag in range(1, min(max_lag + 1, len(x))):
            num = np.sum((x[:-lag] - mu) * (x[lag:] - mu))
            acf_val = num / var
            if acf_val > prev:
                return lag - 1 if lag > 1 else 1
            prev = acf_val

        return max_lag
