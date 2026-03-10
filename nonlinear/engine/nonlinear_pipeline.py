"""
Full automated nonlinear analysis pipeline.

Combines phase-space embedding, fractal metrics, Lyapunov estimation,
entropy, and recurrence analysis into a single ``run()`` call.
"""

import numpy as np

from ..core.timeseries import TimeSeries
from ..core.embedding import PhaseSpace
from ..diagnostics.time_series_diagnostics import TimeSeriesDiagnostics
from ..attractor.recurrence import RecurrenceAnalysis


class NonlinearAnalysisEngine:
    """End-to-end nonlinear diagnostics pipeline.

    Parameters
    ----------
    series : :class:`~nonlinear.core.timeseries.TimeSeries` or array-like
        The input time series.
    """

    def __init__(self, series):
        if isinstance(series, TimeSeries):
            self.ts = series
        else:
            self.ts = TimeSeries(series)
        self._diag = TimeSeriesDiagnostics

    def run(self, emb_dim=4, delay=None, epsilon=None):
        """Execute the full diagnostic pipeline.

        Parameters
        ----------
        emb_dim : int
            Embedding dimension for phase-space reconstruction (default 4).
        delay : int or None
            Embedding delay.  If *None*, estimated automatically via the
            first minimum of the autocorrelation function.
        epsilon : float or None
            Recurrence threshold.  If *None*, set to 10 % of the standard
            deviation of the embedded trajectory distances.

        Returns
        -------
        dict
            Dictionary with keys:

            - ``hurst_rs``  – Hurst exponent (R/S method)
            - ``hurst_dfa`` – Hurst exponent (DFA method)
            - ``higuchi``   – Higuchi fractal dimension
            - ``lyapunov``  – largest Lyapunov exponent (Rosenstein)
            - ``entropy``   – Shannon histogram entropy
            - ``permutation_entropy`` – normalised permutation entropy
            - ``recurrence_rate`` – recurrence rate of the attractor
            - ``determinism`` – RQA determinism measure
        """
        x = self.ts.data
        results = {}

        # Phase-space reconstruction
        ps = PhaseSpace(self.ts)
        if delay is None:
            delay = ps.optimal_delay()
        attractor = ps.embed(emb_dim, delay)

        # Hurst exponents
        if len(x) >= 100:
            results["hurst_rs"] = self._diag.hurst_exponent(x)
        if len(x) >= 32:
            results["hurst_dfa"] = self._diag.hurst_dfa(x)

        # Fractal dimension
        results["higuchi"] = self._diag.higuchi_dimension(x)

        # Lyapunov (from data, Rosenstein)
        results["lyapunov"] = self._diag.lyapunov_rosenstein(
            x, emb_dim=emb_dim, tau=delay
        )

        # Entropy measures
        results["entropy"] = self._diag.complexity(x)
        results["permutation_entropy"] = self._diag.permutation_entropy(
            x, order=min(5, len(x) // 2)
        )

        # Recurrence analysis
        ra = RecurrenceAnalysis(attractor)
        if epsilon is None:
            # Use 10% of mean pairwise distance as threshold
            sample_idx = np.random.default_rng(0).choice(
                len(attractor), size=min(200, len(attractor)), replace=False
            )
            sample = attractor[sample_idx]
            dists = np.linalg.norm(
                sample[:, np.newaxis] - sample[np.newaxis, :], axis=2
            )
            epsilon = 0.1 * np.mean(dists[dists > 0])

        results["recurrence_rate"] = ra.recurrence_rate(epsilon)
        results["determinism"] = ra.determinism(epsilon)

        return results
