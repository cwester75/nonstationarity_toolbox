import numpy as np
from nonlinear.core.map2d import Map2D
from nonlinear.maps2d.map2d import iterate_map
from nonlinear.maps2d.henon_map import HenonMap


def test_map2d_iterate_shape():
    h = HenonMap()
    traj = h.iterate((0.1, 0.1), 100)
    assert traj.shape == (101, 2)  # n+1 points (including initial)


def test_iterate_map_standalone():
    h = HenonMap()
    xs, ys = iterate_map(h, 0.1, 0.1, 100)
    assert len(xs) == 100
    assert len(ys) == 100


def test_map2d_subclass_contract():
    class TestMap(Map2D):
        def step(self, state):
            return (state[0] * 0.5, state[1] * 0.5)

        def jacobian(self, state):
            return np.array([[0.5, 0], [0, 0.5]])

    m = TestMap()
    x, y = m.step((1.0, 1.0))
    assert x == 0.5
    assert y == 0.5

    traj = m.iterate((1.0, 1.0), 5)
    assert traj.shape == (6, 2)
    assert abs(traj[-1, 0] - 1.0 / 32) < 1e-10
