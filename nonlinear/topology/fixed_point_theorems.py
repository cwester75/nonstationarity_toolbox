import numpy as np


def banach_iteration(f, x0, n=100, tol=1e-12):
    """Demonstrate Banach fixed point theorem by iterating a contraction mapping.

    Returns (fixed_point, trajectory, converged, contraction_ratios).
    """
    traj = [np.array(x0, dtype=float)]
    ratios = []
    x = np.array(x0, dtype=float)

    for i in range(n):
        x_new = np.array(f(x), dtype=float)
        traj.append(x_new)

        if i > 0:
            d_new = np.linalg.norm(x_new - x)
            d_old = np.linalg.norm(x - traj[-3])
            if d_old > 0:
                ratios.append(d_new / d_old)

        if np.linalg.norm(x_new - x) < tol:
            return x_new, traj, True, ratios

        x = x_new

    return x, traj, False, ratios


def verify_contraction(f, x_range, grid=1000, h=1e-6):
    """Verify that f is a contraction on the given interval.

    Estimates the Lipschitz constant numerically.
    """
    xs = np.linspace(x_range[0], x_range[1], grid)
    max_ratio = 0.0

    for i in range(grid - 1):
        df = abs(f(xs[i + 1]) - f(xs[i]))
        dx = abs(xs[i + 1] - xs[i])
        if dx > 0:
            ratio = df / dx
            max_ratio = max(max_ratio, ratio)

    return max_ratio, max_ratio < 1.0


def brouwer_demo_1d(f, x_range=(0, 1), grid=10000):
    """Demonstrate Brouwer's fixed point theorem in 1-D.

    For f: [a,b] -> [a,b], there must be a fixed point.
    Returns approximate fixed point locations.
    """
    xs = np.linspace(x_range[0], x_range[1], grid)
    fixed_points = []

    for i in range(grid - 1):
        g0 = f(xs[i]) - xs[i]
        g1 = f(xs[i + 1]) - xs[i + 1]
        if g0 * g1 <= 0:
            # Linear interpolation for zero
            if abs(g1 - g0) > 1e-15:
                x_fp = xs[i] - g0 * (xs[i + 1] - xs[i]) / (g1 - g0)
            else:
                x_fp = 0.5 * (xs[i] + xs[i + 1])
            fixed_points.append(x_fp)

    return fixed_points
