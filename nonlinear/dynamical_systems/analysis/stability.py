import numpy as np

from .fixed_points import jacobian


def classify_fixed_point(system, point):
    """Classify a fixed point by eigenvalues of the Jacobian.

    Parameters
    ----------
    system : AutonomousSystem2D
    point : array-like, shape (2,)
        The fixed point (x, y).

    Returns
    -------
    str
        One of: "stable node", "unstable node", "saddle",
        "stable spiral", "unstable spiral", "center", "undetermined".
    """
    J = jacobian(system, point[0], point[1])
    eigenvalues = np.linalg.eigvals(J)

    r1, r2 = eigenvalues

    # Check for complex eigenvalues (spirals or center)
    if np.iscomplex(r1) or abs(r1.imag) > 1e-10:
        if r1.real < -1e-10:
            return "stable spiral"
        elif r1.real > 1e-10:
            return "unstable spiral"
        else:
            return "center"

    # Real eigenvalues
    r1, r2 = r1.real, r2.real

    if r1 * r2 < 0:
        return "saddle"

    if r1 < 0 and r2 < 0:
        return "stable node"

    if r1 > 0 and r2 > 0:
        return "unstable node"

    return "undetermined"
