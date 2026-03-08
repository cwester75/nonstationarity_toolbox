import numpy as np
import matplotlib.pyplot as plt


def plot_feigenbaum_convergence(deltas, ax=None):
    """Plot convergence of Feigenbaum delta estimates.

    Args:
        deltas: list of delta estimates from estimate_feigenbaum_delta.
        ax: optional matplotlib axes.
    """
    if ax is None:
        fig, ax = plt.subplots()

    indices = np.arange(1, len(deltas) + 1)
    ax.plot(indices, deltas, 'bo-', markersize=6)
    ax.axhline(4.6692, color='red', linewidth=1, linestyle='--',
               label=r'$\delta = 4.6692...$')

    ax.set_xlabel('Period-Doubling Index')
    ax.set_ylabel(r'$\delta$ Estimate')
    ax.set_title('Feigenbaum Delta Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_period_doubling_cascade(doublings, ax=None):
    """Plot period-doubling bifurcation points on the parameter axis.

    Args:
        doublings: list of parameter values where period doublings occur.
        ax: optional matplotlib axes.
    """
    if ax is None:
        fig, ax = plt.subplots()

    for i, r in enumerate(doublings):
        period = 2 ** (i + 1)
        ax.axvline(r, color='blue', linewidth=0.8, alpha=0.7)
        ax.text(r, 0.5 + 0.1 * (i % 3), f'P{period}',
                ha='center', fontsize=8, rotation=90)

    ax.set_xlabel('r')
    ax.set_yticks([])
    ax.set_title('Period-Doubling Cascade')
    return ax
