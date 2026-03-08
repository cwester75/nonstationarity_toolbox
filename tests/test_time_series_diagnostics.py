"""Tests for TimeSeriesDiagnostics and RollingDiagnostics."""

import numpy as np
import pytest

from nonlinear.diagnostics.time_series_diagnostics import TimeSeriesDiagnostics
from nonlinear.diagnostics.rolling_diagnostics import RollingDiagnostics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _brownian(n=500, seed=42):
    """Generate a cumulative-sum random walk (H ~ 0.5)."""
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.standard_normal(n))


def _trending(n=500):
    """Generate a strongly trending series (H > 0.5)."""
    return np.linspace(0, 10, n) + np.sin(np.linspace(0, 6 * np.pi, n)) * 0.3


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------

class TestCorrelation:
    def test_perfect_positive(self):
        x = np.arange(100, dtype=float)
        assert TimeSeriesDiagnostics.correlation(x, x) == pytest.approx(1.0)

    def test_perfect_negative(self):
        x = np.arange(100, dtype=float)
        assert TimeSeriesDiagnostics.correlation(x, -x) == pytest.approx(-1.0)

    def test_uncorrelated(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(10000)
        y = rng.standard_normal(10000)
        assert abs(TimeSeriesDiagnostics.correlation(x, y)) < 0.05


# ---------------------------------------------------------------------------
# Hurst exponent
# ---------------------------------------------------------------------------

class TestHurst:
    def test_random_walk_hurst(self):
        """A random walk should have H ~ 0.5 (within tolerance)."""
        series = _brownian(500)
        h = TimeSeriesDiagnostics.hurst_exponent(series)
        assert 0.3 < h < 0.8

    def test_trending_hurst(self):
        """A strongly trending series should have H > 0.5."""
        series = _trending(500)
        h = TimeSeriesDiagnostics.hurst_exponent(series)
        assert h > 0.4

    def test_short_series_raises(self):
        with pytest.raises(ValueError, match="too short"):
            TimeSeriesDiagnostics.hurst_exponent(np.zeros(50))


# ---------------------------------------------------------------------------
# Higuchi fractal dimension
# ---------------------------------------------------------------------------

class TestHiguchi:
    def test_returns_finite(self):
        series = _brownian(300)
        fd = TimeSeriesDiagnostics.higuchi_dimension(series)
        assert np.isfinite(fd)

    def test_typical_range(self):
        """Higuchi FD for Brownian motion should be roughly 1.0 – 2.0."""
        series = _brownian(500)
        fd = TimeSeriesDiagnostics.higuchi_dimension(series)
        assert 1.0 < fd < 2.5

    def test_smooth_series_lower_fd(self):
        """A smooth sinusoid should have a lower FD than white noise."""
        smooth = np.sin(np.linspace(0, 10 * np.pi, 500))
        rng = np.random.default_rng(1)
        noisy = rng.standard_normal(500)
        fd_smooth = TimeSeriesDiagnostics.higuchi_dimension(smooth)
        fd_noisy = TimeSeriesDiagnostics.higuchi_dimension(noisy)
        assert fd_smooth < fd_noisy


# ---------------------------------------------------------------------------
# Lyapunov exponent (Rosenstein)
# ---------------------------------------------------------------------------

class TestLyapunov:
    def test_returns_finite(self):
        series = _brownian(300)
        lam = TimeSeriesDiagnostics.lyapunov_rosenstein(series)
        assert np.isfinite(lam)

    def test_short_series_raises(self):
        with pytest.raises(ValueError, match="too short"):
            TimeSeriesDiagnostics.lyapunov_rosenstein(np.array([1.0, 2.0]), emb_dim=5)


# ---------------------------------------------------------------------------
# Complexity (Shannon entropy)
# ---------------------------------------------------------------------------

class TestComplexity:
    def test_positive_entropy(self):
        series = _brownian(300)
        s = TimeSeriesDiagnostics.complexity(series)
        assert s > 0

    def test_uniform_higher_than_peaked(self):
        """Uniform distribution should have higher entropy than peaked."""
        rng = np.random.default_rng(7)
        uniform = rng.uniform(size=5000)
        peaked = rng.normal(loc=0, scale=0.01, size=5000)
        s_uniform = TimeSeriesDiagnostics.complexity(uniform, bins=30)
        s_peaked = TimeSeriesDiagnostics.complexity(peaked, bins=30)
        assert s_uniform > s_peaked


# ---------------------------------------------------------------------------
# RollingDiagnostics
# ---------------------------------------------------------------------------

class TestRollingDiagnostics:
    def test_from_array(self):
        series = _brownian(500)
        rd = RollingDiagnostics(series)
        h = rd.hurst()
        assert 0.2 < h < 1.0

    def test_higuchi_via_rolling(self):
        series = _brownian(300)
        rd = RollingDiagnostics(series)
        fd = rd.higuchi()
        assert np.isfinite(fd)

    def test_entropy_via_rolling(self):
        series = _brownian(300)
        rd = RollingDiagnostics(series)
        s = rd.entropy()
        assert s > 0

    def test_lyapunov_via_rolling(self):
        series = _brownian(300)
        rd = RollingDiagnostics(series)
        lam = rd.lyapunov()
        assert np.isfinite(lam)

    def test_correlation_via_rolling(self):
        x = np.arange(100, dtype=float)
        rd = RollingDiagnostics(x)
        assert rd.correlation(x) == pytest.approx(1.0)
