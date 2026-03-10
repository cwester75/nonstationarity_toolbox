"""Tests for the NonlinearAnalysisEngine pipeline."""

import numpy as np

from nonlinear.core.timeseries import TimeSeries
from nonlinear.engine.nonlinear_pipeline import NonlinearAnalysisEngine


def _brownian(n=500, seed=42):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.standard_normal(n))


class TestNonlinearPipeline:
    def test_run_returns_all_keys(self):
        series = TimeSeries(_brownian(500))
        engine = NonlinearAnalysisEngine(series)
        results = engine.run()

        expected_keys = {
            "hurst_rs", "hurst_dfa", "higuchi", "lyapunov",
            "entropy", "permutation_entropy",
            "recurrence_rate", "determinism",
        }
        assert expected_keys == set(results.keys())

    def test_run_values_finite(self):
        series = TimeSeries(_brownian(500))
        engine = NonlinearAnalysisEngine(series)
        results = engine.run()

        for key, value in results.items():
            assert np.isfinite(value), f"{key} is not finite: {value}"

    def test_run_from_array(self):
        """Engine should accept a plain array."""
        engine = NonlinearAnalysisEngine(_brownian(300))
        results = engine.run()
        assert "higuchi" in results

    def test_hurst_values_reasonable(self):
        series = TimeSeries(_brownian(500))
        results = NonlinearAnalysisEngine(series).run()
        assert 0.1 < results["hurst_rs"] < 1.5
        assert 0.1 < results["hurst_dfa"] < 1.5

    def test_permutation_entropy_bounded(self):
        series = TimeSeries(_brownian(500))
        results = NonlinearAnalysisEngine(series).run()
        # Normalised permutation entropy should be in [0, 1]
        assert 0.0 <= results["permutation_entropy"] <= 1.0

    def test_recurrence_rate_bounded(self):
        series = TimeSeries(_brownian(300))
        results = NonlinearAnalysisEngine(series).run()
        assert 0.0 <= results["recurrence_rate"] <= 1.0
        assert 0.0 <= results["determinism"] <= 1.0
