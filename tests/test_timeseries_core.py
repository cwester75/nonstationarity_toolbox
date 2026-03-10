"""Tests for TimeSeries and PhaseSpace core modules."""

import numpy as np
import pytest

from nonlinear.core.timeseries import TimeSeries
from nonlinear.core.embedding import PhaseSpace


# ---------------------------------------------------------------------------
# TimeSeries
# ---------------------------------------------------------------------------

class TestTimeSeries:
    def test_length(self):
        ts = TimeSeries([1, 2, 3, 4, 5])
        assert ts.length == 5
        assert len(ts) == 5

    def test_normalize(self):
        ts = TimeSeries([10, 20, 30, 40, 50])
        ts.normalize()
        assert abs(np.mean(ts.data)) < 1e-10
        assert abs(np.std(ts.data) - 1.0) < 1e-10

    def test_normalize_zero_variance(self):
        ts = TimeSeries([5.0, 5.0, 5.0])
        ts.normalize()
        assert np.all(ts.data == 0.0)

    def test_normalize_returns_self(self):
        ts = TimeSeries([1, 2, 3])
        result = ts.normalize()
        assert result is ts

    def test_difference(self):
        ts = TimeSeries([10, 12, 15, 11])
        diff = ts.difference(lag=1)
        np.testing.assert_array_equal(diff.data, [2, 3, -4])

    def test_difference_lag2(self):
        ts = TimeSeries([1, 2, 4, 7])
        diff = ts.difference(lag=2)
        np.testing.assert_array_equal(diff.data, [3, 5])

    def test_rolling_window(self):
        ts = TimeSeries(np.arange(10))
        windows = ts.rolling_window(3)
        assert windows.shape == (8, 3)
        np.testing.assert_array_equal(windows[0], [0, 1, 2])
        np.testing.assert_array_equal(windows[-1], [7, 8, 9])

    def test_rolling_window_too_large(self):
        ts = TimeSeries([1, 2])
        with pytest.raises(ValueError):
            ts.rolling_window(5)

    def test_subseries(self):
        ts = TimeSeries(np.arange(20), name="test")
        sub = ts.subseries(5, 10)
        assert sub.length == 5
        np.testing.assert_array_equal(sub.data, [5, 6, 7, 8, 9])

    def test_repr(self):
        ts = TimeSeries([1, 2, 3], name="spy")
        assert "spy" in repr(ts)
        assert "3" in repr(ts)


# ---------------------------------------------------------------------------
# PhaseSpace / Embedding
# ---------------------------------------------------------------------------

class TestPhaseSpace:
    def test_embed_shape(self):
        x = np.arange(100, dtype=float)
        ps = PhaseSpace(x)
        emb = ps.embed(dimension=3, delay=1)
        assert emb.shape == (98, 3)

    def test_embed_with_delay(self):
        x = np.arange(100, dtype=float)
        ps = PhaseSpace(x)
        emb = ps.embed(dimension=3, delay=5)
        # M = 100 - (3-1)*5 = 90
        assert emb.shape == (90, 3)
        # First vector should be [0, 5, 10]
        np.testing.assert_array_equal(emb[0], [0, 5, 10])

    def test_embed_from_timeseries(self):
        ts = TimeSeries(np.arange(50, dtype=float))
        ps = PhaseSpace(ts)
        emb = ps.embed(dimension=2, delay=1)
        assert emb.shape == (49, 2)

    def test_embed_too_short(self):
        x = np.array([1.0, 2.0, 3.0])
        ps = PhaseSpace(x)
        with pytest.raises(ValueError, match="too short"):
            ps.embed(dimension=5, delay=2)

    def test_optimal_delay_sine(self):
        """Sinusoid should produce a reasonable delay."""
        t = np.linspace(0, 40 * np.pi, 2000)
        x = np.sin(t)
        ps = PhaseSpace(x)
        tau = ps.optimal_delay(max_lag=100)
        # Quarter period = 2000 / (40/2) / 4 = 25
        assert 1 <= tau <= 50

    def test_optimal_delay_constant(self):
        x = np.ones(100)
        ps = PhaseSpace(x)
        tau = ps.optimal_delay()
        assert tau == 1
