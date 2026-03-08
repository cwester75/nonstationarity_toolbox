import numpy as np
from nonlinear.core.maps import LogisticMap
from nonlinear.diagnostics.stability import classify_fixed_point_1d, stability_multiplier
from nonlinear.diagnostics.fixed_points import find_fixed_points
from nonlinear.diagnostics.fixed_points_2d import classify_fixed_point_2d
from nonlinear.maps2d.henon_map import HenonMap


def test_logistic_fixed_point_stable():
    """r=2.5: fixed point at x* = 1 - 1/r = 0.6 is stable."""
    m = LogisticMap(2.5)
    classification = classify_fixed_point_1d(m, 0.6)
    assert classification == "stable"


def test_logistic_fixed_point_unstable():
    """r=4: fixed point at x* = 0.75 is unstable."""
    m = LogisticMap(4)
    classification = classify_fixed_point_1d(m, 0.75)
    assert classification == "unstable"


def test_henon_fixed_point_classification():
    """Henon map fixed points should be saddle type at a=1.4."""
    h = HenonMap(a=1.4, b=0.3)
    fps = h.fixed_points()
    J = h.jacobian(fps[0])
    classification, eigenvalues, tr, det = classify_fixed_point_2d(J)
    assert classification in ("saddle", "unstable node", "unstable spiral")
