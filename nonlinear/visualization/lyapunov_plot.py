import numpy as np
import matplotlib.pyplot as plt
from nonlinear.core.maps import LogisticMap
from nonlinear.core.trajectory import iterate_map
from nonlinear.diagnostics.lyapunov import lyapunov_exponent


def plot_lyapunov_vs_parameter(map_class, r_range, grid=500, n=1000, x0=0.5, ax=None):
    """Plot Lyapunov exponent as a function of map parameter."""
    r_values = np.linspace(r_range[0], r_range[1], grid)
    les = []

    for r in r_values:
        m = map_class(r)
        traj = iterate_map(m, x0, n)
        les.append(lyapunov_exponent(m, traj))

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(r_values, les, linewidth=0.5)
    ax.axhline(0, color='red', linewidth=0.5, linestyle='--')
    ax.set_xlabel('Parameter')
    ax.set_ylabel('Lyapunov Exponent')
    ax.set_title('Lyapunov Exponent vs Parameter')
    return ax


def plot_lyapunov_convergence(map_obj, x0=0.5, max_n=5000, ax=None):
    """Plot convergence of Lyapunov exponent with trajectory length."""
    traj = iterate_map(map_obj, x0, max_n)
    ns = np.arange(100, max_n, 100)
    les = []

    for n in ns:
        les.append(lyapunov_exponent(map_obj, traj[:n]))

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(ns, les)
    ax.set_xlabel('Trajectory Length')
    ax.set_ylabel('Lyapunov Exponent')
    ax.set_title('Lyapunov Exponent Convergence')
    return ax
