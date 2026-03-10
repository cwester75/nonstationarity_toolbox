import numpy as np


def detect_limit_cycle(trajectory, section_coord=0, section_value=0.0,
                       min_crossings=6, variance_threshold=1e-3):
    """Detect a limit cycle using a Poincare section crossing method.

    Monitors crossings of a coordinate plane and checks whether
    the crossing points converge, indicating a periodic orbit.

    Parameters
    ----------
    trajectory : ndarray, shape (N, 2)
        Simulated trajectory.
    section_coord : int
        Which coordinate to use for the section (0 for x, 1 for y).
    section_value : float
        Value of the section coordinate at the crossing plane.
    min_crossings : int
        Minimum number of crossings required.
    variance_threshold : float
        Maximum variance of crossing positions to declare a cycle.

    Returns
    -------
    bool
        True if a limit cycle is detected.
    """
    points = trajectory
    other_coord = 1 - section_coord
    crossings = []

    for i in range(1, len(points)):
        prev = points[i - 1, section_coord] - section_value
        curr = points[i, section_coord] - section_value

        # Detect upward crossing (negative to non-negative)
        if prev < 0 and curr >= 0:
            crossings.append(points[i, other_coord])

    if len(crossings) < min_crossings:
        return False

    crossings = np.array(crossings)
    variance = np.var(crossings)
    return bool(variance < variance_threshold)
