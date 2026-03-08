import numpy as np


def newton_step(f, df, x):
    """Single Newton iteration: x_{n+1} = x - f(x)/f'(x)."""
    return x - f(x) / df(x)


def newton_iterate(f, df, x0, n=100, tol=1e-12):
    """Run Newton's method, returning trajectory and convergence info."""
    traj = [x0]
    x = x0
    for i in range(n):
        dx = f(x) / df(x)
        x = x - dx
        traj.append(x)
        if abs(dx) < tol:
            return traj, True, i + 1
    return traj, False, n


def newton_basins(f, df, x_range, grid=500, n=100, tol=1e-6):
    """Compute basin of attraction for Newton's method on a 1-D grid."""
    xs = np.linspace(x_range[0], x_range[1], grid)
    roots = []
    basin = np.zeros(grid, dtype=int)

    for i, x0 in enumerate(xs):
        traj, converged, _ = newton_iterate(f, df, x0, n, tol)
        if converged:
            root = traj[-1]
            found = False
            for j, r in enumerate(roots):
                if abs(root - r) < tol * 10:
                    basin[i] = j
                    found = True
                    break
            if not found:
                roots.append(root)
                basin[i] = len(roots) - 1
        else:
            basin[i] = -1

    return xs, basin, roots
