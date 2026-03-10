"""Visualization functions for Hamiltonian systems (Chapter 4).

Provides plotting routines for:
    - Phase-space trajectories
    - Poincaré surfaces of section
    - Energy conservation diagnostics
    - Action-angle space
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_phase_space(
    q_traj: np.ndarray,
    p_traj: np.ndarray,
    q_index: int = 0,
    p_index: int = 0,
    ax: plt.Axes | None = None,
    title: str = "Phase Space",
    **kwargs,
) -> plt.Axes:
    """Plot a phase-space trajectory (q vs p).

    Parameters
    ----------
    q_traj, p_traj : ndarray, shape (steps, ndof)
    q_index, p_index : int
        Which degree of freedom to plot.
    ax : matplotlib Axes, optional
    title : str
    **kwargs
        Passed to ax.plot().

    Returns
    -------
    matplotlib Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    defaults = {"linewidth": 0.3, "color": "steelblue"}
    defaults.update(kwargs)
    ax.plot(q_traj[:, q_index], p_traj[:, p_index], **defaults)
    ax.set_xlabel(f"$q_{{{q_index + 1}}}$")
    ax.set_ylabel(f"$p_{{{p_index + 1}}}$")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    return ax


def plot_poincare_section(
    section_points: np.ndarray,
    col_x: int = 0,
    col_y: int = 1,
    ax: plt.Axes | None = None,
    title: str = "Poincaré Section",
    **kwargs,
) -> plt.Axes:
    """Plot a Poincaré surface of section.

    Parameters
    ----------
    section_points : ndarray, shape (n, d)
        Output from compute_poincare_section.
    col_x, col_y : int
        Column indices to plot.
    ax : matplotlib Axes, optional
    title : str
    **kwargs
        Passed to ax.scatter().

    Returns
    -------
    matplotlib Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    defaults = {"s": 0.5, "color": "black", "marker": "."}
    defaults.update(kwargs)
    ax.scatter(section_points[:, col_x], section_points[:, col_y], **defaults)
    ax.set_xlabel(f"coord {col_x + 1}")
    ax.set_ylabel(f"coord {col_y + 1}")
    ax.set_title(title)
    return ax


def plot_energy_drift(
    system,
    q_traj: np.ndarray,
    p_traj: np.ndarray,
    dt: float = 0.01,
    ax: plt.Axes | None = None,
    title: str = "Energy Conservation",
) -> plt.Axes:
    """Plot relative energy drift over time.

    Parameters
    ----------
    system : HamiltonianSystem
    q_traj, p_traj : ndarray, shape (steps, ndof)
    dt : float
        Time step (for labeling the x-axis).
    ax : matplotlib Axes, optional
    title : str

    Returns
    -------
    matplotlib Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    energies = np.array([
        system.energy(q_traj[i], p_traj[i])
        for i in range(len(q_traj))
    ])
    e0 = energies[0]
    if abs(e0) > 1e-15:
        relative = (energies - e0) / abs(e0)
    else:
        relative = energies - e0
    t = np.arange(len(energies)) * dt
    ax.plot(t, relative, linewidth=0.5)
    ax.set_xlabel("Time")
    ax.set_ylabel("$\\Delta E / E_0$")
    ax.set_title(title)
    ax.ticklabel_format(style="sci", axis="y", scilimits=(-3, 3))
    return ax


def plot_multi_trajectory_phase_space(
    trajectories: list[tuple[np.ndarray, np.ndarray]],
    q_index: int = 0,
    p_index: int = 0,
    ax: plt.Axes | None = None,
    title: str = "Phase Space",
    colors: list[str] | None = None,
) -> plt.Axes:
    """Plot multiple phase-space trajectories on a single axis.

    Parameters
    ----------
    trajectories : list of (q_traj, p_traj) tuples
    q_index, p_index : int
    ax : matplotlib Axes, optional
    title : str
    colors : list of str, optional

    Returns
    -------
    matplotlib Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    if colors is None:
        cmap = plt.cm.tab10
        colors = [cmap(i % 10) for i in range(len(trajectories))]
    for (q, p), c in zip(trajectories, colors):
        ax.plot(q[:, q_index], p[:, p_index], linewidth=0.3, color=c)
    ax.set_xlabel(f"$q_{{{q_index + 1}}}$")
    ax.set_ylabel(f"$p_{{{p_index + 1}}}$")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    return ax
