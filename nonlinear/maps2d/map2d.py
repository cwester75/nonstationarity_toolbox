"""Re-export Map2D and provide standalone iterate_map for 2-D maps."""
import numpy as np
from nonlinear.core.map2d import Map2D  # noqa: F401


def iterate_map(map_obj, x0, y0, n):
    """Iterate a 2-D map n times, returning separate x and y arrays.

    Convenience wrapper around Map2D.iterate() that accepts and returns
    separate x, y coordinates for compatibility with plotting functions.
    """
    traj = map_obj.iterate((x0, y0), n)
    return traj[1:, 0], traj[1:, 1]
