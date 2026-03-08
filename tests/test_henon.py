import numpy as np
from nonlinear.maps2d.henon_map import HenonMap
from nonlinear.diagnostics.lyapunov_spectrum import lyapunov_spectrum


def test_henon_step():
    h = HenonMap(a=1.4, b=0.3)
    x, y = h.step((0.0, 0.0))
    assert abs(x - 1.0) < 1e-10
    assert abs(y - 0.0) < 1e-10


def test_henon_fixed_points():
    h = HenonMap(a=1.4, b=0.3)
    fps = h.fixed_points()
    assert len(fps) == 2
    for fp in fps:
        x_new, y_new = h.step(fp)
        assert abs(x_new - fp[0]) < 1e-10
        assert abs(y_new - fp[1]) < 1e-10


def test_henon_chaotic():
    h = HenonMap(a=1.4, b=0.3)
    exponents = lyapunov_spectrum(h, (0.1, 0.1), n=5000)
    assert exponents[0] > 0  # largest exponent positive
    assert exponents[1] < 0  # second exponent negative
