"""Tests for recurrence analysis."""

import numpy as np
import pytest

from nonlinear.attractor.recurrence import RecurrenceAnalysis
from nonlinear.core.embedding import PhaseSpace


class TestRecurrenceMatrix:
    def test_shape(self):
        ps = np.random.default_rng(0).standard_normal((50, 3))
        ra = RecurrenceAnalysis(ps)
        R = ra.recurrence_matrix(epsilon=1.0)
        assert R.shape == (50, 50)

    def test_symmetric(self):
        ps = np.random.default_rng(0).standard_normal((30, 2))
        ra = RecurrenceAnalysis(ps)
        R = ra.recurrence_matrix(epsilon=2.0)
        np.testing.assert_array_equal(R, R.T)

    def test_diagonal_all_true(self):
        """Distance from a point to itself is 0, which is < any epsilon > 0."""
        ps = np.random.default_rng(0).standard_normal((20, 2))
        ra = RecurrenceAnalysis(ps)
        R = ra.recurrence_matrix(epsilon=0.01)
        assert np.all(np.diag(R))

    def test_large_epsilon_all_recurrent(self):
        ps = np.random.default_rng(0).standard_normal((15, 2))
        ra = RecurrenceAnalysis(ps)
        R = ra.recurrence_matrix(epsilon=1e6)
        assert R.all()


class TestRecurrenceRate:
    def test_range(self):
        ps = np.random.default_rng(0).standard_normal((40, 3))
        ra = RecurrenceAnalysis(ps)
        rr = ra.recurrence_rate(epsilon=1.5)
        assert 0.0 <= rr <= 1.0

    def test_large_epsilon_gives_one(self):
        ps = np.random.default_rng(0).standard_normal((20, 2))
        ra = RecurrenceAnalysis(ps)
        rr = ra.recurrence_rate(epsilon=1e6)
        assert rr == pytest.approx(1.0)


class TestDeterminism:
    def test_range(self):
        ps = np.random.default_rng(0).standard_normal((40, 3))
        ra = RecurrenceAnalysis(ps)
        det = ra.determinism(epsilon=2.0)
        assert 0.0 <= det <= 1.0

    def test_periodic_signal_high_determinism(self):
        """A periodic signal embedded should have high determinism."""
        t = np.linspace(0, 20 * np.pi, 500)
        x = np.sin(t)
        ps = PhaseSpace(x)
        emb = ps.embed(dimension=3, delay=5)
        ra = RecurrenceAnalysis(emb)
        det = ra.determinism(epsilon=0.3)
        assert det > 0.3
