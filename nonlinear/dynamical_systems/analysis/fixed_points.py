import numpy as np


def jacobian(system, x, y, eps=1e-6):
    """Compute the Jacobian of the vector field via finite differences.

    Parameters
    ----------
    system : AutonomousSystem2D
    x, y : float
        Point at which to evaluate.
    eps : float
        Finite-difference step size.

    Returns
    -------
    ndarray, shape (2, 2)
    """
    fx, gx = system.vector_field(x, y)

    fx_dx, _ = system.vector_field(x + eps, y)
    fx_dy, _ = system.vector_field(x, y + eps)
    _, gx_dx = system.vector_field(x + eps, y)
    _, gx_dy = system.vector_field(x, y + eps)

    dfdx = (fx_dx - fx) / eps
    dfdy = (fx_dy - fx) / eps
    dgdx = (gx_dx - gx) / eps
    dgdy = (gx_dy - gx) / eps

    return np.array([[dfdx, dfdy],
                     [dgdx, dgdy]])


def find_fixed_points(system, initial_guesses, max_iter=50, tol=1e-10):
    """Locate fixed points of the system using Newton's method.

    Solves f(x,y) = 0, g(x,y) = 0.

    Parameters
    ----------
    system : AutonomousSystem2D
    initial_guesses : list of tuple
        Starting points [(x0, y0), ...].
    max_iter : int
        Maximum Newton iterations per guess.
    tol : float
        Convergence tolerance on the residual norm.

    Returns
    -------
    list of ndarray
        Converged fixed points (duplicates removed).
    """
    fixed_points = []

    for guess in initial_guesses:
        x, y = float(guess[0]), float(guess[1])

        for _ in range(max_iter):
            fx, fy = system.vector_field(x, y)
            residual = np.array([fx, fy])
            if np.linalg.norm(residual) < tol:
                break

            J = jacobian(system, x, y)
            try:
                delta = np.linalg.solve(J, -residual)
            except np.linalg.LinAlgError:
                break

            x += delta[0]
            y += delta[1]

        fx, fy = system.vector_field(x, y)
        if np.linalg.norm([fx, fy]) < tol * 100:
            fixed_points.append(np.array([x, y]))

    # Remove duplicates
    unique = []
    for fp in fixed_points:
        is_dup = False
        for u in unique:
            if np.linalg.norm(fp - u) < 1e-6:
                is_dup = True
                break
        if not is_dup:
            unique.append(fp)

    return unique
