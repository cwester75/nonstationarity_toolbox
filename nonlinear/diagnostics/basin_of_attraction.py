import numpy as np


def basin_of_attraction_2d(map2d, attractors, x_range, y_range,
                            grid=200, max_iter=500, tol=0.1):
    """Compute basin of attraction for a 2-D map.

    attractors: list of attractor states [(x, y), ...]
    Returns grid of attractor indices.
    """
    xs = np.linspace(x_range[0], x_range[1], grid)
    ys = np.linspace(y_range[0], y_range[1], grid)
    basin = np.full((grid, grid), -1, dtype=int)

    for i, x0 in enumerate(xs):
        for j, y0 in enumerate(ys):
            state = np.array([x0, y0])
            for _ in range(max_iter):
                try:
                    state = np.array(map2d.step(state))
                except (OverflowError, FloatingPointError):
                    break
                if np.any(np.abs(state) > 1e10):
                    break

                for k, att in enumerate(attractors):
                    if np.linalg.norm(state - np.array(att)) < tol:
                        basin[j, i] = k
                        break
                if basin[j, i] >= 0:
                    break

    return xs, ys, basin


def basin_of_attraction_1d(map_obj, attractors, x_range, grid=1000,
                            max_iter=500, tol=1e-4):
    """Compute basin of attraction for a 1-D map."""
    xs = np.linspace(x_range[0], x_range[1], grid)
    basin = np.full(grid, -1, dtype=int)

    for i, x0 in enumerate(xs):
        x = x0
        for _ in range(max_iter):
            x = map_obj.f(x)
            if abs(x) > 1e10:
                break
            for k, att in enumerate(attractors):
                if abs(x - att) < tol:
                    basin[i] = k
                    break
            if basin[i] >= 0:
                break

    return xs, basin
