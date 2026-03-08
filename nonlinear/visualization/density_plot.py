import numpy as np
import matplotlib.pyplot as plt
from nonlinear.statistics.invariant_density import estimate_density


def plot_invariant_density(traj, bins=200, analytical_func=None, ax=None):
    """Plot estimated invariant density, optionally overlaid with analytical curve.

    Args:
        traj: 1-D trajectory array.
        bins: number of histogram bins.
        analytical_func: if provided, overlay the analytical density.
        ax: optional matplotlib axes.
    """
    centers, counts = estimate_density(traj, bins)

    if ax is None:
        fig, ax = plt.subplots()

    ax.bar(centers, counts, width=centers[1] - centers[0],
           alpha=0.6, label='Estimated')

    if analytical_func is not None:
        x_fine = np.linspace(centers[0], centers[-1], 500)
        ax.plot(x_fine, analytical_func(x_fine), 'r-',
                linewidth=1.5, label='Analytical')
        ax.legend()

    ax.set_xlabel('x')
    ax.set_ylabel('Density')
    ax.set_title('Invariant Density')
    return ax
