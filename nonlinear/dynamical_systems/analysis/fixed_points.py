import numpy as np
from scipy.optimize import root


def jacobian(system, x, y, eps=1e-6):
    """Compute the Jacobian of the vector field via central differences.

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
    fx1 = system.vector_field(x + eps, y)
    fx2 = system.vector_field(x - eps, y)

    fy1 = system.vector_field(x, y + eps)
    fy2 = system.vector_field(x, y - eps)

    dfdx = (fx1[0] - fx2[0]) / (2 * eps)
    dfdy = (fy1[0] - fy2[0]) / (2 * eps)

    dgdx = (fx1[1] - fx2[1]) / (2 * eps)
    dgdy = (fy1[1] - fy2[1]) / (2 * eps)

    return np.array([[dfdx, dfdy],
                     [dgdx, dgdy]])


def find_fixed_point(system, guess):
    """Find a single fixed point starting from *guess* using SciPy root.

    Parameters
    ----------
    system : AutonomousSystem2D
    guess : array-like, shape (2,)

    Returns
    -------
    ndarray or None
        The fixed point, or None if the solver did not converge.
    """
    def F(v):
        return system.vector_field(v[0], v[1])

    sol = root(F, guess)

    if sol.success:
        return sol.x

    return None


def find_fixed_points(system, initial_guesses):
    """Locate fixed points of the system using SciPy root finding.

    Solves f(x,y) = 0, g(x,y) = 0.

    Parameters
    ----------
    system : AutonomousSystem2D
    initial_guesses : list of tuple
        Starting points [(x0, y0), ...].

    Returns
    -------
    list of ndarray
        Converged fixed points (duplicates removed).
    """
    fixed_points = []

    for guess in initial_guesses:
        fp = find_fixed_point(system, guess)
        if fp is not None:
            fixed_points.append(fp)

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
