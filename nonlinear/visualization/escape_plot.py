import numpy as np
import matplotlib.pyplot as plt


def plot_escape_time(xs, times, ax=None):
    """Plot escape time as a function of initial condition.

    Args:
        xs: array of initial conditions from escape_time_grid.
        times: array of escape times.
        ax: optional matplotlib axes.
    """
    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(xs, times, ',', markersize=0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('Escape Time')
    ax.set_title('Escape Time Diagram')
    return ax


def plot_repeller(repeller_points, ax=None):
    """Plot the chaotic repeller as a set of points on the x-axis.

    Args:
        repeller_points: array of x values on the repeller.
        ax: optional matplotlib axes.
    """
    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(repeller_points, np.zeros_like(repeller_points), '|',
            markersize=10, color='black')
    ax.set_xlabel('x')
    ax.set_yticks([])
    ax.set_title(f'Chaotic Repeller ({len(repeller_points)} points)')
    return ax


def plot_survival_probability(times, survival, kappa=None, ax=None):
    """Plot survival probability P(t) on a semilogy scale.

    Args:
        times: time array from survival_probability.
        survival: P(t) array.
        kappa: estimated escape rate (shown as fit line if provided).
        ax: optional matplotlib axes.
    """
    if ax is None:
        fig, ax = plt.subplots()

    mask = survival > 0
    ax.semilogy(times[mask], survival[mask], 'b-', linewidth=0.8, label='P(t)')

    if kappa is not None:
        t_fit = times[mask]
        ax.semilogy(t_fit, np.exp(-kappa * t_fit), 'r--',
                    linewidth=1, label=f'exp(-{kappa:.4f}t)')
        ax.legend()

    ax.set_xlabel('Time')
    ax.set_ylabel('P(t)')
    ax.set_title('Survival Probability')
    return ax
