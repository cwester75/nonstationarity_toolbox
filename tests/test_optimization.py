import pytest
import numpy as np
from nonlinear.optimization.newton_map import newton_step, newton_iterate
from nonlinear.optimization.newton_complex import newton_complex_step, newton_complex_iterate
from nonlinear.optimization.newton_nd import newton_nd_iterate, numerical_jacobian


def test_newton_1d_quadratic():
    f = lambda x: x ** 2 - 4
    df = lambda x: 2 * x
    traj, converged, iters = newton_iterate(f, df, 3.0)
    assert converged
    assert abs(traj[-1] - 2.0) < 1e-10


def test_newton_complex_cube_roots():
    f = lambda z: z ** 3 - 1
    df = lambda z: 3 * z ** 2
    z, iters, converged = newton_complex_iterate(f, df, complex(0.5, 0.5))
    assert converged
    # Should converge to one of the cube roots of unity
    assert abs(z ** 3 - 1) < 1e-8


def test_newton_nd_2d_system():
    # Solve x^2 + y^2 = 1, x - y = 0 => x = y = 1/sqrt(2)
    def F(v):
        return np.array([v[0] ** 2 + v[1] ** 2 - 1, v[0] - v[1]])

    def J(v):
        return np.array([
            [2 * v[0], 2 * v[1]],
            [1.0, -1.0]
        ])

    x, traj, converged, iters = newton_nd_iterate(F, J, [1.0, 0.5])
    assert converged
    assert abs(x[0] - x[1]) < 1e-10
    assert abs(x[0] ** 2 + x[1] ** 2 - 1) < 1e-10


def test_numerical_jacobian():
    def F(x):
        return np.array([x[0] ** 2, x[0] * x[1]])

    J = numerical_jacobian(F, [2.0, 3.0])
    # dF1/dx1 = 2*x1 = 4, dF1/dx2 = 0
    # dF2/dx1 = x2 = 3, dF2/dx2 = x1 = 2
    assert abs(J[0, 0] - 4.0) < 1e-5
    assert abs(J[0, 1]) < 1e-5
    assert abs(J[1, 0] - 3.0) < 1e-5
    assert abs(J[1, 1] - 2.0) < 1e-5


def test_newton_step_zero_derivative():
    f = lambda x: x ** 2
    df = lambda x: 2 * x
    with pytest.raises(ZeroDivisionError):
        newton_step(f, df, 0.0)


def test_newton_iterate_zero_derivative_graceful():
    # x^3 has df=0 at x=0; starting at 0 should not converge, not crash
    f = lambda x: x ** 3
    df = lambda x: 3 * x ** 2
    traj, converged, iters = newton_iterate(f, df, 0.0)
    assert not converged


def test_newton_complex_step_zero_derivative():
    f = lambda z: z ** 2
    df = lambda z: 2 * z
    with pytest.raises(ZeroDivisionError):
        newton_complex_step(f, df, 0 + 0j)


def test_newton_complex_iterate_zero_derivative_graceful():
    f = lambda z: z ** 3
    df = lambda z: 3 * z ** 2
    z, iters, converged = newton_complex_iterate(f, df, 0 + 0j)
    assert not converged
