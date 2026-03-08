import numpy as np
import matplotlib.pyplot as plt
from nonlinear.bifurcation.bifurcation_diagram import generate_bifurcation


def plot_bifurcation(r_range, iterations=1000, last=100, ax=None):
    """Plot a bifurcation diagram for the logistic map."""
    r_values = np.linspace(r_range[0], r_range[1], 2000)
    rs, xs = generate_bifurcation(r_values, iterations, last)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(rs, xs, ',', color='black', markersize=0.1)
    ax.set_xlabel('r')
    ax.set_ylabel('x')
    ax.set_title('Bifurcation Diagram')
    return ax


def plot_bifurcation_with_lyapunov(r_range, ax=None):
    """Plot bifurcation diagram alongside Lyapunov exponent."""
    from nonlinear.core.maps import LogisticMap
    from nonlinear.core.trajectory import iterate_map
    from nonlinear.diagnostics.lyapunov import lyapunov_exponent

    if ax is None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    else:
        ax1, ax2 = ax

    r_values = np.linspace(r_range[0], r_range[1], 1000)

    # Bifurcation
    rs, xs = generate_bifurcation(r_values, 1000, 100)
    ax1.plot(rs, xs, ',', color='black', markersize=0.1)
    ax1.set_ylabel('x')
    ax1.set_title('Bifurcation Diagram')

    # Lyapunov
    les = []
    for r in r_values:
        m = LogisticMap(r)
        traj = iterate_map(m, 0.5, 1000)
        les.append(lyapunov_exponent(m, traj))

    ax2.plot(r_values, les, linewidth=0.5)
    ax2.axhline(0, color='red', linewidth=0.5)
    ax2.set_xlabel('r')
    ax2.set_ylabel('Lyapunov exponent')

    return ax1, ax2
