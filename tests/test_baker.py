import numpy as np
from nonlinear.maps2d.baker_map import BakerMap


def test_baker_left():
    b = BakerMap()
    x, y = b.step((0.25, 0.5))
    assert abs(x - 0.5) < 1e-10
    assert abs(y - 0.25) < 1e-10


def test_baker_right():
    b = BakerMap()
    x, y = b.step((0.75, 0.5))
    assert abs(x - 0.5) < 1e-10
    assert abs(y - 0.75) < 1e-10


def test_baker_jacobian():
    b = BakerMap()
    J = b.jacobian((0.25, 0.5))
    assert abs(np.linalg.det(J) - 1.0) < 1e-10  # area-preserving
