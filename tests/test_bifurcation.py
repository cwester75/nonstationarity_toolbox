import numpy as np
from nonlinear.bifurcation.bifurcation_diagram import generate_bifurcation
from nonlinear.bifurcation.feigenbaum import (
    find_period_doubling_points,
    estimate_feigenbaum_delta,
    logistic_map_func,
)
from nonlinear.bifurcation.chaos_detector import is_chaotic
from nonlinear.core.maps import LogisticMap


def test_bifurcation_diagram_output_sizes():
    r_values = np.linspace(3.0, 4.0, 10)
    rs, xs = generate_bifurcation(r_values, iterations=200, last=20)
    assert len(rs) == len(xs)
    assert len(rs) == 10 * 20  # 10 r-values * last=20 points each


def test_bifurcation_values_in_range():
    r_values = np.linspace(2.5, 4.0, 50)
    rs, xs = generate_bifurcation(r_values, iterations=500, last=50)
    assert all(0 <= x <= 1 for x in xs)


def test_feigenbaum_delta_estimates():
    doublings = find_period_doubling_points(logistic_map_func, (2.8, 3.6))
    if len(doublings) >= 3:
        deltas = estimate_feigenbaum_delta(doublings)
        assert deltas is not None
        assert len(deltas) >= 1


def test_is_chaotic_at_r4():
    m = LogisticMap(4)
    chaotic, le = is_chaotic(m, n=2000)
    assert chaotic
    assert le > 0


def test_not_chaotic_at_r2():
    m = LogisticMap(2.0)
    chaotic, le = is_chaotic(m, n=2000)
    assert not chaotic
    assert le < 0
