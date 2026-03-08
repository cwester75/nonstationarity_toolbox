import numpy as np


def escape_time(map_func, x0, max_iter=1000, bound=1e6):
    """Compute escape time for an initial condition."""
    x = x0
    for i in range(max_iter):
        x = map_func(x)
        if abs(x) > bound:
            return i
    return max_iter


def escape_time_grid(map_func, x_range, grid=1000, max_iter=500, bound=1e6):
    """Compute escape times over a grid of initial conditions."""
    xs = np.linspace(x_range[0], x_range[1], grid)
    times = np.zeros(grid, dtype=int)
    for i, x0 in enumerate(xs):
        times[i] = escape_time(map_func, x0, max_iter, bound)
    return xs, times


def repeller_set(map_func, x_range, grid=5000, max_iter=500, bound=1e6):
    """Identify points on the chaotic repeller (those that never escape)."""
    xs = np.linspace(x_range[0], x_range[1], grid)
    repeller = []
    for x0 in xs:
        if escape_time(map_func, x0, max_iter, bound) == max_iter:
            repeller.append(x0)
    return np.array(repeller)
