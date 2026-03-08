import numpy as np
from nonlinear.core.orbit_tools import orbit_distance, detect_periodicity


def test_orbit_distance():
    assert orbit_distance([1, 2], [1, 2]) == 0.0
    assert abs(orbit_distance([0, 0], [3, 4]) - 5.0) < 1e-10


def test_detect_periodicity_period1():
    # Constant sequence has period 1
    traj = [0.5] * 100
    assert detect_periodicity(traj) == 1


def test_detect_periodicity_period2():
    traj = [0.3, 0.7] * 50
    assert detect_periodicity(traj) == 2


def test_detect_periodicity_none():
    # Chaotic trajectory should not detect low periodicity
    np.random.seed(42)
    traj = list(np.random.rand(100))
    result = detect_periodicity(traj, tol=1e-10)
    assert result is None
