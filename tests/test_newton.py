import numpy as np
from nonlinear.optimization.newton_map import newton_iterate


def test_newton_sqrt2():
    """Newton's method to find sqrt(2) via x^2 - 2 = 0."""
    f = lambda x: x**2 - 2
    df = lambda x: 2*x
    traj, converged, iters = newton_iterate(f, df, 1.0, n=50)
    assert converged
    assert abs(traj[-1] - np.sqrt(2)) < 1e-10


def test_newton_cubic():
    """Newton's method for x^3 - x = 0."""
    f = lambda x: x**3 - x
    df = lambda x: 3*x**2 - 1
    traj, converged, iters = newton_iterate(f, df, 0.6, n=50)
    assert converged
    # Should find root at 0 or ±1
    root = traj[-1]
    assert min(abs(root), abs(root - 1), abs(root + 1)) < 1e-10
