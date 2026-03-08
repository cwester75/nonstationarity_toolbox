import numpy as np


def estimate_density(traj, bins=200):
    """Estimate invariant density from a trajectory using histogram."""
    counts, edges = np.histogram(traj, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, counts


def logistic_invariant_density(x):
    """Analytical invariant density for logistic map at r=4: 1/(pi*sqrt(x*(1-x)))."""
    x = np.asarray(x, dtype=float)
    result = np.zeros_like(x)
    mask = (x > 0) & (x < 1)
    result[mask] = 1.0 / (np.pi * np.sqrt(x[mask] * (1 - x[mask])))
    return result


def density_error(traj, analytical_func, bins=200):
    """Compute L2 error between estimated and analytical invariant density."""
    centers, estimated = estimate_density(traj, bins)
    analytical = analytical_func(centers)
    dx = centers[1] - centers[0]
    l2 = np.sqrt(np.sum((estimated - analytical) ** 2) * dx)
    return l2
