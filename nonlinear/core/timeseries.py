"""
Core time-series object for nonlinear analysis.

Handles preprocessing, normalization, windowing, and differencing.
Serves as the standard input to all diagnostics and pipeline components.
"""

import numpy as np


class TimeSeries:
    """A time series with metadata and preprocessing utilities.

    Parameters
    ----------
    data : array-like
        Raw time-series values.
    dt : float
        Sampling interval (default 1.0).
    name : str
        Label for the series (default ``"unknown"``).
    """

    def __init__(self, data, dt=1.0, name="unknown"):
        self.data = np.asarray(data, dtype=float)
        self.dt = dt
        self.name = name

    @property
    def length(self):
        return len(self.data)

    # ---- preprocessing ----------------------------------------------------

    def normalize(self):
        """Z-score normalise the series in-place.

        Returns *self* so calls can be chained.
        """
        mu = np.mean(self.data)
        sigma = np.std(self.data)
        if sigma > 0:
            self.data = (self.data - mu) / sigma
        else:
            self.data = self.data - mu
        return self

    def difference(self, lag=1):
        """Return the lag-*lag* difference series as a new :class:`TimeSeries`."""
        diff = self.data[lag:] - self.data[:-lag]
        return TimeSeries(diff, dt=self.dt, name=f"{self.name}_diff{lag}")

    def rolling_window(self, window):
        """Yield overlapping segments of length *window*.

        Returns a 2-D array of shape ``(N - window + 1, window)``.
        """
        N = self.length
        if window > N:
            raise ValueError(f"Window ({window}) exceeds series length ({N})")
        return np.lib.stride_tricks.sliding_window_view(self.data, window)

    def subseries(self, start, end):
        """Return a slice ``[start:end]`` as a new :class:`TimeSeries`."""
        return TimeSeries(self.data[start:end], dt=self.dt,
                          name=f"{self.name}[{start}:{end}]")

    def __len__(self):
        return self.length

    def __repr__(self):
        return f"TimeSeries(name={self.name!r}, length={self.length}, dt={self.dt})"
