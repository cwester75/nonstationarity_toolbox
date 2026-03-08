import numpy as np
from nonlinear.core.maps import LogisticMap
from nonlinear.core.trajectory import iterate_map


def test_iterate_returns_array():
    m = LogisticMap(3.5)
    traj = iterate_map(m, 0.5, 10)
    assert isinstance(traj, np.ndarray)
    assert len(traj) == 10


def test_iterate_zero_steps():
    m = LogisticMap(3.5)
    traj = iterate_map(m, 0.5, 0)
    assert len(traj) == 0


def test_iterate_values_in_range():
    m = LogisticMap(3.5)
    traj = iterate_map(m, 0.2, 1000)
    assert np.all(traj >= 0)
    assert np.all(traj <= 1)
