import numpy as np


def classify_fixed_point_2d(jacobian):
    """Classify a 2-D fixed point from its Jacobian matrix.

    Returns classification string and eigenvalues.
    """
    eigenvalues = np.linalg.eigvals(jacobian)
    mags = np.abs(eigenvalues)
    tr = np.trace(jacobian)
    det = np.linalg.det(jacobian)

    is_complex = np.any(np.abs(eigenvalues.imag) > 1e-10)

    if all(m < 1 for m in mags):
        if is_complex:
            classification = "stable spiral"
        else:
            classification = "stable node"
    elif all(m > 1 for m in mags):
        if is_complex:
            classification = "unstable spiral"
        else:
            classification = "unstable node"
    elif mags[0] < 1 < mags[1] or mags[1] < 1 < mags[0]:
        classification = "saddle"
    else:
        classification = "marginal"

    return classification, eigenvalues, tr, det


def find_fixed_points_2d(map2d, x_range, y_range, grid=50, tol=1e-6, max_iter=100):
    """Find fixed points of a 2-D map by Newton's method on a grid of initial guesses."""
    fixed_points = []

    for x0 in np.linspace(x_range[0], x_range[1], grid):
        for y0 in np.linspace(y_range[0], y_range[1], grid):
            state = np.array([x0, y0])
            for _ in range(max_iter):
                mapped = np.array(map2d.step(state))
                residual = mapped - state
                if np.linalg.norm(residual) < tol:
                    # Check if this is a new fixed point
                    is_new = True
                    for fp in fixed_points:
                        if np.linalg.norm(state - fp) < tol * 10:
                            is_new = False
                            break
                    if is_new:
                        fixed_points.append(state.copy())
                    break

                J = map2d.jacobian(state) - np.eye(2)
                try:
                    delta = np.linalg.solve(J, -residual)
                except np.linalg.LinAlgError:
                    break
                state = state + delta

    return fixed_points
