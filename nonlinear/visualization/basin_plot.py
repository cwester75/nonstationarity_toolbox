import numpy as np
import matplotlib.pyplot as plt


def plot_basin_2d(xs, ys, basin, ax=None):
    """Plot 2-D basin of attraction as a colored grid.

    Args:
        xs, ys: coordinate arrays from basin_of_attraction_2d.
        basin: 2-D integer array of attractor indices.
        ax: optional matplotlib axes.
    """
    if ax is None:
        fig, ax = plt.subplots()

    ax.imshow(basin, extent=[xs[0], xs[-1], ys[0], ys[-1]],
              origin='lower', cmap='Set1', interpolation='nearest', aspect='auto')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Basin of Attraction')
    return ax


def plot_basin_1d(xs, basin, ax=None):
    """Plot 1-D basin of attraction as a colored strip.

    Args:
        xs: coordinate array from basin_of_attraction_1d.
        basin: 1-D integer array of attractor indices.
        ax: optional matplotlib axes.
    """
    if ax is None:
        fig, ax = plt.subplots()

    colors = basin.astype(float)
    colors[basin == -1] = np.nan
    ax.scatter(xs, np.zeros_like(xs), c=colors, cmap='Set1', s=1, marker='|')
    ax.set_xlabel('x')
    ax.set_yticks([])
    ax.set_title('Basin of Attraction')
    return ax
