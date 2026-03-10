import numpy as np


def detect_limit_cycle(trajectory, section_coord=0, section_value=0.0,
                       min_crossings=5, variance_threshold=1e-3):
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
    other_coord = 1 - section_coord
    crossings = []

    for i in range(1, len(trajectory)):
        prev = trajectory[i - 1, section_coord] - section_value
        curr = trajectory[i, section_coord] - section_value

        # Detect upward crossing (negative to non-negative)
        if prev < 0 and curr >= 0:
            crossings.append(trajectory[i, other_coord])

    if len(crossings) < min_crossings:
        return False

    # Use only the latter half of crossings (after transients)
    stable_crossings = crossings[len(crossings) // 2:]
    if len(stable_crossings) < 3:
        return False

    variance = np.var(stable_crossings)
    return bool(variance < variance_threshold)
