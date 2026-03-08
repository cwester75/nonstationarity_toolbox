import numpy as np


def iterate_map(map_obj, x0, n):
    """Iterate a 1-D map n times, returning the trajectory as a numpy array."""
    traj = np.empty(n)
    x = x0

    for i in range(n):
        x = map_obj.f(x)
        traj[i] = x

    return traj
