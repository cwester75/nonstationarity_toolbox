"""
Rolling-window adapter for time-series diagnostics.

Provides a convenience wrapper that converts a rolling window (or any
indexable container) into a NumPy array and delegates to
:class:`TimeSeriesDiagnostics`.
"""

import numpy as np

from .time_series_diagnostics import TimeSeriesDiagnostics


class RollingDiagnostics:
    """Wrap a rolling window and expose diagnostic methods.

    Parameters
    ----------
    window : indexable container
        Any object supporting ``window[i]`` and ``window.count``
        (e.g. a QuantConnect ``RollingWindow``), or a plain list / array.
    """

    def __init__(self, window):
        self.window = window

    def to_array(self):
        """Convert the window to a NumPy array (oldest-first)."""
        if isinstance(self.window, np.ndarray):
            return self.window
        if hasattr(self.window, "count"):
            return np.array(
                [self.window[i] for i in range(self.window.count)][::-1]
            )
        return np.asarray(self.window, dtype=float)

    def hurst(self):
        """Hurst exponent of the windowed series."""
        return TimeSeriesDiagnostics.hurst_exponent(self.to_array())

    def higuchi(self, kmax=10):
        """Higuchi fractal dimension of the windowed series."""
        return TimeSeriesDiagnostics.higuchi_dimension(self.to_array(), kmax=kmax)

    def lyapunov(self, emb_dim=3, tau=1, max_iter=10):
        """Largest Lyapunov exponent (Rosenstein) of the windowed series."""
        return TimeSeriesDiagnostics.lyapunov_rosenstein(
            self.to_array(), emb_dim=emb_dim, tau=tau, max_iter=max_iter
        )

    def entropy(self, bins=20):
        """Shannon entropy (complexity) of the windowed series."""
        return TimeSeriesDiagnostics.complexity(self.to_array(), bins=bins)

    def correlation(self, other):
        """Pearson correlation between this window and *other*."""
        return TimeSeriesDiagnostics.correlation(self.to_array(), np.asarray(other))
