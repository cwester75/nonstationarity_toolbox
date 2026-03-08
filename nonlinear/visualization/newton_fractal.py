import numpy as np
import matplotlib.pyplot as plt
from nonlinear.optimization.newton_complex import newton_fractal_grid


def plot_newton_fractal(f, df, x_range=(-2, 2), y_range=(-2, 2),
                         grid=500, max_iter=50, ax=None):
    """Plot a Newton fractal in the complex plane."""
    root_grid, iter_grid, roots = newton_fractal_grid(
        f, df, x_range, y_range, grid, max_iter
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    ax.imshow(root_grid, extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
              origin='lower', cmap='Set1', interpolation='nearest')
    ax.set_xlabel('Re(z)')
    ax.set_ylabel('Im(z)')
    ax.set_title(f'Newton Fractal ({len(roots)} roots)')
    return ax


def plot_newton_convergence(f, df, x_range=(-2, 2), y_range=(-2, 2),
                             grid=500, max_iter=50, ax=None):
    """Plot Newton fractal colored by convergence speed."""
    _, iter_grid, _ = newton_fractal_grid(
        f, df, x_range, y_range, grid, max_iter
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    ax.imshow(iter_grid, extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
              origin='lower', cmap='hot', interpolation='nearest')
    ax.set_xlabel('Re(z)')
    ax.set_ylabel('Im(z)')
    ax.set_title('Newton Fractal (Convergence Speed)')
    return ax
