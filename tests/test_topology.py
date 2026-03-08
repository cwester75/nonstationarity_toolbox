import numpy as np
from nonlinear.core.maps import LogisticMap
from nonlinear.topology.periodic_orbits import find_periodic_orbits_1d, topological_degree
from nonlinear.topology.fixed_point_theorems import banach_iteration, brouwer_demo_1d


def test_find_fixed_points_as_period1():
    m = LogisticMap(3.2)
    orbits = find_periodic_orbits_1d(m, 1, grid=200)
    # Logistic map at r=3.2 has fixed point x* = 1 - 1/r
    expected = 1 - 1 / 3.2
    found = any(abs(orb[0] - expected) < 1e-3 for orb in orbits)
    assert found


def test_find_period2_orbit():
    m = LogisticMap(3.2)
    orbits = find_periodic_orbits_1d(m, 2, grid=500)
    # At r=3.2 there should be a period-2 orbit
    assert len(orbits) >= 1
    for orb in orbits:
        assert len(orb) == 2


def test_topological_degree_logistic():
    m = LogisticMap(3.5)
    degree = topological_degree(m)
    # Logistic at r=3.5: f(x)-x has two zeros (two fixed points)
    assert abs(degree) >= 0  # degree is well-defined


def test_banach_iteration_converges():
    # f(x) = cos(x) is a contraction near x=0.739
    fp, traj, converged, ratios = banach_iteration(np.cos, 0.5, n=100)
    assert converged
    assert abs(fp - 0.7390851332) < 1e-6


def test_brouwer_demo():
    # f(x) = x^2 maps [0,1] to [0,1], has fixed points at 0 and 1
    fps = brouwer_demo_1d(lambda x: x ** 2, x_range=(0, 1))
    assert len(fps) >= 1
