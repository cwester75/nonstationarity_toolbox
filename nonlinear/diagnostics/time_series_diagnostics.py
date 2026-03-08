"""
Nonlinear time-series diagnostics for financial and scientific data.

Provides correlation, Hurst exponent (R/S method), Higuchi fractal dimension,
Lyapunov exponent (Rosenstein method), and Shannon entropy complexity measures.
"""

import numpy as np
from numpy.linalg import norm
from scipy.stats import entropy


class TimeSeriesDiagnostics:
    """
    Collection of nonlinear time-series diagnostics useful for financial data.
    Designed to work with price arrays or rolling window conversions.
    """

    # ---------------------------------------------------------
    # CORRELATION
    # ---------------------------------------------------------

    @staticmethod
    def correlation(x, y):
        """Pearson correlation between two arrays."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        return np.corrcoef(x, y)[0, 1]

    # ---------------------------------------------------------
    # HURST EXPONENT (R/S METHOD)
    # ---------------------------------------------------------

    @staticmethod
    def hurst_exponent(series):
        """Estimate the Hurst exponent via the rescaled range (R/S) method.

        Returns a value in approximately [0, 1]:
        - H < 0.5  => mean-reverting
        - H ~ 0.5  => random walk
        - H > 0.5  => trending / persistent

        Parameters
        ----------
        series : array-like
            Time series of at least 100 observations.

        Returns
        -------
        float
            Estimated Hurst exponent.
        """
        ts = np.asarray(series, dtype=float)
        N = len(ts)

        if N < 100:
            raise ValueError("Series too short for Hurst estimation (need >= 100)")

        lags = range(2, 100)
        tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]

        poly = np.polyfit(np.log(list(lags)), np.log(tau), 1)
        return poly[0] * 2.0

    # ---------------------------------------------------------
    # HIGUCHI FRACTAL DIMENSION
    # ---------------------------------------------------------

    @staticmethod
    def higuchi_dimension(series, kmax=10):
        """Compute the Higuchi fractal dimension of a time series.

        Parameters
        ----------
        series : array-like
            Input time series.
        kmax : int
            Maximum delay/interval parameter (default 10).

        Returns
        -------
        float
            Estimated fractal dimension (typically 1.0 – 2.0).
        """
        x = np.asarray(series, dtype=float)
        N = len(x)

        L = []
        k_values = range(1, kmax)

        for k in k_values:
            Lk = []
            for m in range(k):
                length = 0.0
                n_max = int((N - m - 1) / k)
                if n_max < 1:
                    continue
                for i in range(1, n_max + 1):
                    length += abs(x[m + i * k] - x[m + (i - 1) * k])
                norm_factor = (N - 1) / (k * n_max * k)
                Lk.append(length * norm_factor)
            if Lk:
                L.append(np.mean(Lk))

        logL = np.log(L)
        logk = np.log(1.0 / np.array(list(k_values[: len(L)])))

        slope = np.polyfit(logk, logL, 1)[0]
        return slope

    # ---------------------------------------------------------
    # LYAPUNOV EXPONENT (ROSENSTEIN METHOD)
    # ---------------------------------------------------------

    @staticmethod
    def lyapunov_rosenstein(series, emb_dim=3, tau=1, max_iter=10):
        """Estimate the largest Lyapunov exponent using Rosenstein's method.

        Embeds the series in phase space, finds nearest neighbours, and
        measures divergence over time.

        Parameters
        ----------
        series : array-like
            Input time series.
        emb_dim : int
            Embedding dimension (default 3).
        tau : int
            Time delay for embedding (default 1).
        max_iter : int
            Number of iterations to track divergence (default 10).

        Returns
        -------
        float
            Estimated largest Lyapunov exponent.
        """
        x = np.asarray(series, dtype=float)
        N = len(x)

        # Phase-space embedding
        M = N - (emb_dim - 1) * tau
        if M < 2:
            raise ValueError("Series too short for the requested embedding")

        embedded = np.array([x[i : i + emb_dim * tau : tau] for i in range(M)])

        neighbors = np.zeros(M, dtype=int)

        for i in range(M):
            diff = embedded - embedded[i]
            dist = np.linalg.norm(diff, axis=1)
            dist[i] = np.inf
            neighbors[i] = np.argmin(dist)

        divergence = []
        for k in range(1, max_iter):
            div = []
            for i in range(M - k):
                j = neighbors[i]
                if j + k < M:
                    d = norm(embedded[i + k] - embedded[j + k])
                    div.append(np.log(d + 1e-8))
            if div:
                divergence.append(np.mean(div))

        if len(divergence) < 2:
            return 0.0

        slope = np.polyfit(range(len(divergence)), divergence, 1)[0]
        return slope

    # ---------------------------------------------------------
    # COMPLEXITY (SHANNON ENTROPY)
    # ---------------------------------------------------------

    @staticmethod
    def complexity(series, bins=20):
        """Shannon entropy of the histogram of the series.

        Higher values indicate more complex / random behaviour.

        Parameters
        ----------
        series : array-like
            Input time series.
        bins : int
            Number of histogram bins (default 20).

        Returns
        -------
        float
            Shannon entropy (nats).
        """
        series = np.asarray(series, dtype=float)
        hist, _ = np.histogram(series, bins=bins, density=True)
        hist = hist + 1e-12
        return entropy(hist)
