import numpy as np


def orbit_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


def detect_periodicity(traj, tol=1e-5):
    for p in range(1, len(traj) // 2):
        if abs(traj[-1] - traj[-1 - p]) < tol:
            return p

    return None
