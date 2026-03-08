import numpy as np
import matplotlib.pyplot as plt


def plot_melnikov(t0_values, M_values, ax=None):
    """Plot the Melnikov function M(t0).

    Sign changes indicate homoclinic intersections (chaos).

    Args:
        t0_values: array of t0 parameter values.
        M_values: array of Melnikov integral values from melnikov_scan.
        ax: optional matplotlib axes.
    """
    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(t0_values, M_values, 'b-', linewidth=1)
    ax.axhline(0, color='red', linewidth=0.5, linestyle='--')

    # Mark zero crossings
    M = np.array(M_values)
    for i in range(len(M) - 1):
        if M[i] * M[i + 1] < 0:
            # Linear interpolation for zero location
            t_zero = t0_values[i] - M[i] * (t0_values[i + 1] - t0_values[i]) / (M[i + 1] - M[i])
            ax.plot(t_zero, 0, 'ro', markersize=6)

    ax.set_xlabel(r'$t_0$')
    ax.set_ylabel(r'$M(t_0)$')
    ax.set_title('Melnikov Function')
    ax.grid(True, alpha=0.3)
    return ax
