"""Tests for stochastic baseline generators."""

import numpy as np

from nonlinear.stochastic.random_walk import RandomWalk
from nonlinear.stochastic.brownian import BrownianMotion


class TestRandomWalk:
    def test_length(self):
        rw = RandomWalk(seed=42)
        path = rw.generate(500)
        assert len(path) == 500

    def test_starts_at_zero(self):
        rw = RandomWalk(seed=0)
        path = rw.generate(100)
        assert path[0] == 0.0

    def test_reproducible(self):
        p1 = RandomWalk(seed=7).generate(100)
        p2 = RandomWalk(seed=7).generate(100)
        np.testing.assert_array_equal(p1, p2)

    def test_sigma_scales_variance(self):
        p_small = RandomWalk(sigma=0.1, seed=0).generate(10000)
        p_large = RandomWalk(sigma=10.0, seed=0).generate(10000)
        # Increments of large-sigma walk should have higher std
        diff_small = np.diff(p_small)
        diff_large = np.diff(p_large)
        assert np.std(diff_large) > np.std(diff_small) * 10


class TestBrownianMotion:
    def test_length(self):
        bm = BrownianMotion(seed=42)
        path = bm.generate(300)
        assert len(path) == 300

    def test_starts_at_zero(self):
        bm = BrownianMotion(seed=0)
        path = bm.generate(100)
        assert path[0] == 0.0

    def test_dt_affects_variance(self):
        bm1 = BrownianMotion(sigma=1.0, seed=0)
        bm2 = BrownianMotion(sigma=1.0, seed=0)
        p_small_dt = bm1.generate(5000, dt=0.001)
        p_large_dt = bm2.generate(5000, dt=1.0)
        # Larger dt => larger increments
        assert np.std(np.diff(p_large_dt)) > np.std(np.diff(p_small_dt))

    def test_reproducible(self):
        p1 = BrownianMotion(seed=3).generate(200)
        p2 = BrownianMotion(seed=3).generate(200)
        np.testing.assert_array_equal(p1, p2)
