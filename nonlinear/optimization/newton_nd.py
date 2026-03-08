import numpy as np


def newton_nd_step(F, J, x):
    """Single Newton step in N dimensions.

    F: function returning N-vector
    J: function returning NxN Jacobian matrix
    x: current N-vector
    """
    return x - np.linalg.solve(J(x), F(x))


def newton_nd_iterate(F, J, x0, max_iter=100, tol=1e-10):
    """Run Newton's method in N dimensions.

    Returns (solution, trajectory, converged, iterations).
    """
    x = np.array(x0, dtype=float)
    traj = [x.copy()]

    for i in range(max_iter):
        Fx = F(x)
        if np.linalg.norm(Fx) < tol:
            return x, traj, True, i

        Jx = J(x)
        try:
            dx = np.linalg.solve(Jx, Fx)
        except np.linalg.LinAlgError:
            return x, traj, False, i

        x = x - dx
        traj.append(x.copy())

    return x, traj, False, max_iter


def numerical_jacobian(F, x, h=1e-8):
    """Compute Jacobian numerically via finite differences."""
    x = np.array(x, dtype=float)
    n = len(x)
    Fx = F(x)
    m = len(Fx)
    J = np.zeros((m, n))

    for j in range(n):
        x_plus = x.copy()
        x_plus[j] += h
        J[:, j] = (F(x_plus) - Fx) / h

    return J
