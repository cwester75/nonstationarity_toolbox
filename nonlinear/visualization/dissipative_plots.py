"""Visualization functions for dissipative dynamical systems (Chapter 5).

Provides plotting routines for:
    - Phase portraits with vector fields and fixed points
    - Trajectory time series
    - Lyapunov exponent convergence
    - Lyapunov spectrum bar charts
    - Attractor visualization
    - Hopf bifurcation diagrams
    - Phase-space contraction rate
    - Combined chapter summary figure
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from ..dynamical_systems.systems.base_system import AutonomousSystem2D
from ..dynamical_systems.simulation.integrator import simulate
from ..dynamical_systems.analysis.fixed_points import find_fixed_points, jacobian
from ..dynamical_systems.analysis.stability import classify_fixed_point
from ..dynamical_systems.analysis.lyapunov_flow import (
    flow_lyapunov_convergence,
    flow_lyapunov_spectrum,
)
from ..dynamical_systems.analysis.attractors import phase_space_contraction
from .autonomous_phase import (
    plot_vector_field,
    plot_trajectory,
    plot_fixed_points,
    _STABILITY_COLORS,
)


def plot_dissipative_phase_portrait(
    system: AutonomousSystem2D,
    trajectories: list[np.ndarray] | None = None,
    x_range: tuple[float, float] = (-3.0, 3.0),
    y_range: tuple[float, float] = (-3.0, 3.0),
    fixed_point_guesses: list[tuple[float, float]] | None = None,
    ax: plt.Axes | None = None,
    title: str = "Phase Portrait",
) -> plt.Axes:
    """Phase portrait with vector field, trajectories, and stability-colored fixed points.

    Parameters
    ----------
    system : AutonomousSystem2D
    trajectories : list of ndarray, optional
        Pre-computed trajectories, each shape (N, 2).
    x_range, y_range : tuple of float
    fixed_point_guesses : list of (float, float), optional
    ax : matplotlib Axes, optional
    title : str

    Returns
    -------
    matplotlib Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    plot_vector_field(system, x_range, y_range, ax=ax)

    if trajectories is not None:
        cmap = plt.cm.tab10
        for i, traj in enumerate(trajectories):
            ax.plot(
                traj[:, 0], traj[:, 1],
                linewidth=0.6, color=cmap(i % 10), alpha=0.8,
            )

    if fixed_point_guesses is not None:
        fps = find_fixed_points(system, fixed_point_guesses)
        classifications = [classify_fixed_point(system, fp) for fp in fps]
        plot_fixed_points(fps, classifications, ax=ax)

    ax.set_title(title)
    return ax


def plot_dissipative_time_series(
    trajectory: np.ndarray,
    dt: float = 0.01,
    labels: tuple[str, str] = ("x", "y"),
    ax: plt.Axes | None = None,
    title: str = "Time Series",
) -> plt.Axes:
    """Plot x(t) and y(t) coordinate time series.

    Parameters
    ----------
    trajectory : ndarray, shape (N, 2)
    dt : float
    labels : tuple of str
    ax : matplotlib Axes, optional
    title : str

    Returns
    -------
    matplotlib Axes
    """
    if ax is None:
        _, ax = plt.subplots()

    t = np.arange(len(trajectory)) * dt
    ax.plot(t, trajectory[:, 0], linewidth=0.5, label=labels[0])
    ax.plot(t, trajectory[:, 1], linewidth=0.5, label=labels[1], linestyle="--")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.legend(fontsize=8)
    return ax


def plot_flow_lyapunov_convergence(
    system: AutonomousSystem2D,
    x0: float,
    y0: float,
    dt: float = 0.01,
    steps: int = 30000,
    transient: int = 1000,
    ax: plt.Axes | None = None,
    title: str = "Lyapunov Exponent Convergence",
) -> plt.Axes:
    """Plot the running Lyapunov exponent estimate for a flow.

    Parameters
    ----------
    system : AutonomousSystem2D
    x0, y0 : float
    dt : float
    steps : int
    transient : int
    ax : matplotlib Axes, optional
    title : str

    Returns
    -------
    matplotlib Axes
    """
    if ax is None:
        _, ax = plt.subplots()

    times, estimates = flow_lyapunov_convergence(
        system, x0, y0, dt=dt, steps=steps, transient=transient,
    )

    ax.plot(times, estimates, linewidth=0.8, color="steelblue")
    ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
    if len(estimates) > 0:
        final = estimates[-1]
        ax.axhline(
            y=final, color="crimson", linewidth=0.5, linestyle=":",
            label=f"$\\lambda = {final:.4f}$",
        )
        ax.legend(fontsize=8)
    ax.set_xlabel("Time")
    ax.set_ylabel("$\\lambda(t)$")
    ax.set_title(title)
    return ax


def plot_lyapunov_spectrum_bar(
    spectrum: np.ndarray,
    ax: plt.Axes | None = None,
    title: str = "Lyapunov Spectrum",
) -> plt.Axes:
    """Bar chart of the Lyapunov spectrum.

    Parameters
    ----------
    spectrum : ndarray
        Lyapunov exponents, sorted descending.
    ax : matplotlib Axes, optional
    title : str

    Returns
    -------
    matplotlib Axes
    """
    if ax is None:
        _, ax = plt.subplots()

    n = len(spectrum)
    colors = ["crimson" if v > 0 else "steelblue" for v in spectrum]
    bars = ax.bar(range(n), spectrum, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xticks(range(n))
    ax.set_xticklabels([f"$\\lambda_{{{i + 1}}}$" for i in range(n)])
    ax.set_ylabel("Exponent value")
    ax.set_title(title)
    return ax


def plot_contraction_field(
    system: AutonomousSystem2D,
    x_range: tuple[float, float] = (-3.0, 3.0),
    y_range: tuple[float, float] = (-3.0, 3.0),
    n_grid: int = 50,
    ax: plt.Axes | None = None,
    title: str = "Phase-Space Contraction Rate",
) -> plt.Axes:
    """Heatmap of the divergence div(f) showing contraction/expansion regions.

    Parameters
    ----------
    system : AutonomousSystem2D
    x_range, y_range : tuple of float
    n_grid : int
    ax : matplotlib Axes, optional
    title : str

    Returns
    -------
    matplotlib Axes
    """
    if ax is None:
        _, ax = plt.subplots()

    xs = np.linspace(x_range[0], x_range[1], n_grid)
    ys = np.linspace(y_range[0], y_range[1], n_grid)
    X, Y = np.meshgrid(xs, ys)
    div = np.zeros_like(X)

    for i in range(n_grid):
        for j in range(n_grid):
            div[i, j] = phase_space_contraction(system, X[i, j], Y[i, j])

    im = ax.pcolormesh(X, Y, div, cmap="RdBu_r", shading="auto")
    ax.figure.colorbar(im, ax=ax, label="$\\nabla \\cdot f$")
    ax.contour(X, Y, div, levels=[0], colors="black", linewidths=0.8)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    return ax


def plot_hopf_diagram(
    system_factory,
    parameter_values: np.ndarray,
    x0: float = 0.0,
    y0: float = 0.1,
    dt: float = 0.01,
    steps: int = 20000,
    transient: int = 5000,
    ax: plt.Axes | None = None,
    title: str = "Hopf Bifurcation Diagram",
) -> plt.Axes:
    """Bifurcation diagram showing amplitude vs parameter near a Hopf point.

    For each parameter value, simulates the system and records the
    min/max of the x-coordinate in the post-transient tail.

    Parameters
    ----------
    system_factory : callable
        ``system_factory(mu) -> AutonomousSystem2D``
    parameter_values : array-like
    x0, y0 : float
    dt : float
    steps : int
    transient : int
    ax : matplotlib Axes, optional
    title : str

    Returns
    -------
    matplotlib Axes
    """
    if ax is None:
        _, ax = plt.subplots()

    parameter_values = np.asarray(parameter_values, dtype=float)

    for mu in parameter_values:
        system = system_factory(mu)
        traj = simulate(system, x0, y0, steps=steps, dt=dt)
        tail = traj[transient:]
        xmin, xmax = tail[:, 0].min(), tail[:, 0].max()
        ax.plot(
            [mu, mu], [xmin, xmax],
            color="steelblue", linewidth=0.3, marker=".", markersize=0.5,
        )

    ax.set_xlabel("Parameter $\\mu$")
    ax.set_ylabel("$x$")
    ax.set_title(title)
    return ax


def plot_chapter5_summary(
    system: AutonomousSystem2D,
    x0: float,
    y0: float,
    dt: float = 0.01,
    steps: int = 30000,
    transient: int = 3000,
    x_range: tuple[float, float] = (-3.0, 3.0),
    y_range: tuple[float, float] = (-3.0, 3.0),
    fixed_point_guesses: list[tuple[float, float]] | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Generate a 6-panel summary figure for Chapter 5.

    Panels:
        1. Phase portrait (vector field + trajectory + fixed points)
        2. Trajectory time series
        3. Lyapunov exponent convergence
        4. Lyapunov spectrum bar chart
        5. Phase-space contraction rate
        6. Attractor close-up (post-transient trajectory)

    Parameters
    ----------
    system : AutonomousSystem2D
    x0, y0 : float
    dt : float
    steps : int
    transient : int
    x_range, y_range : tuple of float
    fixed_point_guesses : list of (float, float), optional

    Returns
    -------
    tuple of (Figure, ndarray of Axes)
    """
    traj = simulate(system, x0, y0, steps=steps, dt=dt)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes_flat = axes.ravel()

    # 1. Phase portrait
    plot_dissipative_phase_portrait(
        system, trajectories=[traj],
        x_range=x_range, y_range=y_range,
        fixed_point_guesses=fixed_point_guesses,
        ax=axes_flat[0], title="Phase Portrait",
    )

    # 2. Time series (show first 30% for clarity)
    n_show = max(steps // 3, 2000)
    plot_dissipative_time_series(
        traj[:n_show], dt=dt, ax=axes_flat[1], title="Time Series",
    )

    # 3. Lyapunov convergence
    plot_flow_lyapunov_convergence(
        system, x0, y0, dt=dt,
        steps=min(steps, 20000), transient=min(transient, 500),
        ax=axes_flat[2],
    )

    # 4. Lyapunov spectrum
    spectrum = flow_lyapunov_spectrum(
        system, x0, y0, dt=dt,
        steps=min(steps, 20000), transient=min(transient, 500),
    )
    plot_lyapunov_spectrum_bar(spectrum, ax=axes_flat[3])

    # 5. Contraction rate
    plot_contraction_field(
        system, x_range=x_range, y_range=y_range,
        ax=axes_flat[4],
    )

    # 6. Attractor close-up
    tail = traj[transient:]
    axes_flat[5].plot(
        tail[:, 0], tail[:, 1],
        linewidth=0.3, color="steelblue",
    )
    axes_flat[5].set_xlabel("x")
    axes_flat[5].set_ylabel("y")
    axes_flat[5].set_title("Attractor")
    axes_flat[5].set_aspect("equal", adjustable="datalim")

    fig.suptitle("Chapter 5: Dissipative Dynamics Summary", fontsize=14, y=1.01)
    fig.tight_layout()
    return fig, axes_flat
