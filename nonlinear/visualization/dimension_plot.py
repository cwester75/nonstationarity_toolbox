import numpy as np
import matplotlib.pyplot as plt


def plot_correlation_dimension(epsilons, C_values, dimension=None, ax=None):
    """Log-log plot of correlation integral C(epsilon) vs epsilon.

    Args:
        epsilons: array of epsilon values.
        C_values: array of C(epsilon) values.
        dimension: estimated dimension (shown in title if provided).
        ax: optional matplotlib axes.
    """
    if ax is None:
        fig, ax = plt.subplots()

    mask = np.array(C_values) > 0
    ax.loglog(epsilons[mask], np.array(C_values)[mask], 'o-', markersize=4)

    if dimension is not None:
        ax.set_title(f'Correlation Integral (d = {dimension:.3f})')
    else:
        ax.set_title('Correlation Integral')

    ax.set_xlabel(r'$\epsilon$')
    ax.set_ylabel(r'$C(\epsilon)$')
    ax.grid(True, alpha=0.3)
    return ax


def plot_box_counting(box_sizes, counts, dimension=None, ax=None):
    """Log-log plot of box count N(epsilon) vs 1/epsilon.

    Args:
        box_sizes: array of box sizes.
        counts: array of box counts.
        dimension: estimated capacity dimension (shown in title if provided).
        ax: optional matplotlib axes.
    """
    if ax is None:
        fig, ax = plt.subplots()

    mask = counts > 0
    ax.loglog(1.0 / box_sizes[mask], counts[mask].astype(float), 'o-', markersize=4)

    if dimension is not None:
        ax.set_title(f'Box Counting (D = {dimension:.3f})')
    else:
        ax.set_title('Box Counting')

    ax.set_xlabel(r'$1/\epsilon$')
    ax.set_ylabel(r'$N(\epsilon)$')
    ax.grid(True, alpha=0.3)
    return ax
