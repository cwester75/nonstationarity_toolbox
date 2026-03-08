import numpy as np
from nonlinear.phase.circle_map import CircleMap
from nonlinear.phase.rotation_number import rotation_number


def test_circle_map_no_coupling():
    """With K=0, circle map is pure rotation."""
    omega = 0.3
    cm = CircleMap(omega, K=0)
    theta = 0.0
    theta = cm.f(theta)
    assert abs(theta - omega) < 1e-10


def test_rotation_number_rational():
    """For K=0, rotation number should equal omega."""
    omega = 1.0 / 3.0
    rho = rotation_number(omega, K=0, n=10000)
    assert abs(rho - omega) < 1e-4
