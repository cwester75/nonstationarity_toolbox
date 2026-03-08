import numpy as np


def newton_complex_step(f, df, z):
    """Single Newton iteration in the complex plane.

    Raises ZeroDivisionError if df(z) == 0.
    """
    dfz = df(z)
    if dfz == 0:
        raise ZeroDivisionError(f"Derivative is zero at z={z}")
    return z - f(z) / dfz


def newton_complex_iterate(f, df, z0, max_iter=100, tol=1e-10):
    """Run Newton's method in the complex plane.

    Returns (z, iterations, converged).
    If the derivative is zero, iteration stops and returns (z, i, False).
    """
    z = z0
    for i in range(max_iter):
        dfz = df(z)
        if dfz == 0:
            return z, i, False
        dz = f(z) / dfz
        z = z - dz
        if abs(dz) < tol:
            return z, i + 1, True
    return z, max_iter, False


def newton_fractal_grid(f, df, x_range, y_range, grid=500,
                         max_iter=50, tol=1e-6):
    """Generate Newton fractal data over a complex grid.

    Returns (root_index_grid, iteration_count_grid, roots_found).
    """
    xs = np.linspace(x_range[0], x_range[1], grid)
    ys = np.linspace(y_range[0], y_range[1], grid)

    root_grid = np.full((grid, grid), -1, dtype=int)
    iter_grid = np.zeros((grid, grid), dtype=int)
    roots = []

    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            z0 = complex(x, y)
            z, iters, converged = newton_complex_iterate(f, df, z0, max_iter, tol)
            iter_grid[i, j] = iters

            if converged:
                found = False
                for k, r in enumerate(roots):
                    if abs(z - r) < tol * 100:
                        root_grid[i, j] = k
                        found = True
                        break
                if not found:
                    roots.append(z)
                    root_grid[i, j] = len(roots) - 1

    return root_grid, iter_grid, roots
