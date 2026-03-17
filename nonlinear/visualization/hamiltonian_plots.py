"""Visualization functions for Hamiltonian systems (Chapter 4).

Provides plotting routines for:
    - Phase-space trajectories
    - Poincaré surfaces of section
    - Energy conservation diagnostics
    - Energy level-set contours
    - Time-series of coordinates and momenta
    - Lyapunov exponent convergence
    - Phase-space density evolution (Liouville's theorem)
    - Multi-trajectory overlays
    - Combined chapter summary figure
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from ..hamiltonian.system import HamiltonianSystem
from ..hamiltonian.integrators import stormer_verlet_step


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
    system: HamiltonianSystem,
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


# --- New Chapter 4 visualizations ---


def plot_energy_contours(
    system: HamiltonianSystem,
    q_range: tuple[float, float] = (-3.0, 3.0),
    p_range: tuple[float, float] = (-3.0, 3.0),
    n_grid: int = 200,
    n_levels: int = 20,
    ax: plt.Axes | None = None,
    title: str = "Energy Level Sets",
    dof_index: int = 0,
    **kwargs,
) -> plt.Axes:
    """Plot contours of constant energy H(q, p) = E.

    For systems with ndof > 1, other coordinates are set to zero
    and only the specified degree of freedom is varied.

    Parameters
    ----------
    system : HamiltonianSystem
    q_range, p_range : tuple of float
        Coordinate ranges for the grid.
    n_grid : int
        Grid resolution per axis.
    n_levels : int
        Number of contour levels.
    ax : matplotlib Axes, optional
    title : str
    dof_index : int
        Which degree of freedom to vary (others held at zero).
    **kwargs
        Passed to ax.contour().

    Returns
    -------
    matplotlib Axes
    """
    if ax is None:
        _, ax = plt.subplots()

    q_vals = np.linspace(q_range[0], q_range[1], n_grid)
    p_vals = np.linspace(p_range[0], p_range[1], n_grid)
    Q, P = np.meshgrid(q_vals, p_vals)
    H_grid = np.zeros_like(Q)

    ndof = system.ndof
    for i in range(n_grid):
        for j in range(n_grid):
            q = np.zeros(ndof)
            p = np.zeros(ndof)
            q[dof_index] = Q[i, j]
            p[dof_index] = P[i, j]
            H_grid[i, j] = system.energy(q, p)

    defaults = {"cmap": "coolwarm", "linewidths": 0.6}
    defaults.update(kwargs)
    cs = ax.contour(Q, P, H_grid, levels=n_levels, **defaults)
    ax.clabel(cs, inline=True, fontsize=6, fmt="%.2f")
    ax.set_xlabel(f"$q_{{{dof_index + 1}}}$")
    ax.set_ylabel(f"$p_{{{dof_index + 1}}}$")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    return ax


def plot_time_series(
    q_traj: np.ndarray,
    p_traj: np.ndarray,
    dt: float = 0.01,
    q_indices: list[int] | None = None,
    p_indices: list[int] | None = None,
    ax: plt.Axes | None = None,
    title: str = "Time Series",
) -> plt.Axes:
    """Plot time series of selected coordinates and momenta.

    Parameters
    ----------
    q_traj, p_traj : ndarray, shape (steps, ndof)
    dt : float
        Time step.
    q_indices : list of int, optional
        Which q components to plot. Defaults to all.
    p_indices : list of int, optional
        Which p components to plot. Defaults to none.
    ax : matplotlib Axes, optional
    title : str

    Returns
    -------
    matplotlib Axes
    """
    if ax is None:
        _, ax = plt.subplots()

    steps, ndof = q_traj.shape
    t = np.arange(steps) * dt

    if q_indices is None:
        q_indices = list(range(ndof))
    if p_indices is None:
        p_indices = []

    for i in q_indices:
        ax.plot(t, q_traj[:, i], linewidth=0.5, label=f"$q_{{{i + 1}}}$")
    for i in p_indices:
        ax.plot(t, p_traj[:, i], linewidth=0.5, linestyle="--",
                label=f"$p_{{{i + 1}}}$")

    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.legend(fontsize=8)
    return ax


def plot_lyapunov_convergence(
    system: HamiltonianSystem,
    q0: np.ndarray,
    p0: np.ndarray,
    dt: float = 0.01,
    steps: int = 50000,
    transient: int = 1000,
    renorm_interval: int = 10,
    ax: plt.Axes | None = None,
    title: str = "Lyapunov Exponent Convergence",
) -> plt.Axes:
    """Plot the running estimate of the maximal Lyapunov exponent over time.

    Parameters
    ----------
    system : HamiltonianSystem
    q0, p0 : ndarray, shape (ndof,)
        Initial conditions.
    dt : float
        Integration time step.
    steps : int
        Integration steps after transient.
    transient : int
        Initial steps to discard.
    renorm_interval : int
        Steps between renormalizations.
    ax : matplotlib Axes, optional
    title : str

    Returns
    -------
    matplotlib Axes
    """
    if ax is None:
        _, ax = plt.subplots()

    q0 = np.asarray(q0, dtype=float)
    p0 = np.asarray(p0, dtype=float)
    ndof = system.ndof

    # Discard transient
    q, p = q0.copy(), p0.copy()
    for _ in range(transient):
        q, p = stormer_verlet_step(system, q, p, dt)

    # Perturbation
    delta = 1e-9
    dq = np.zeros(ndof)
    dq[0] = delta
    d0 = delta

    q2, p2 = q + dq, p.copy()

    lyap_sum = 0.0
    n_renorm = 0
    times = []
    estimates = []

    for i in range(steps):
        q, p = stormer_verlet_step(system, q, p, dt)
        q2, p2 = stormer_verlet_step(system, q2, p2, dt)

        if (i + 1) % renorm_interval == 0:
            sep_q = q2 - q
            sep_p = p2 - p
            dist = np.sqrt(np.sum(sep_q**2) + np.sum(sep_p**2))

            if dist > 0:
                lyap_sum += np.log(dist / d0)
                n_renorm += 1
                factor = d0 / dist
                q2 = q + sep_q * factor
                p2 = p + sep_p * factor

                times.append((i + 1) * dt)
                estimates.append(lyap_sum / (n_renorm * renorm_interval * dt))

    times = np.array(times)
    estimates = np.array(estimates)

    ax.plot(times, estimates, linewidth=0.8, color="steelblue")
    ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
    if len(estimates) > 0:
        final = estimates[-1]
        ax.axhline(y=final, color="crimson", linewidth=0.5, linestyle=":",
                   label=f"$\\lambda = {final:.4f}$")
        ax.legend(fontsize=8)
    ax.set_xlabel("Time")
    ax.set_ylabel("$\\lambda(t)$")
    ax.set_title(title)
    return ax


def plot_phase_density(
    system: HamiltonianSystem,
    q_center: np.ndarray,
    p_center: np.ndarray,
    spread: float = 0.1,
    n_particles: int = 500,
    dt: float = 0.01,
    n_snapshots: int = 4,
    steps_per_snapshot: int = 500,
    q_index: int = 0,
    p_index: int = 0,
    seed: int = 42,
) -> tuple[plt.Figure, np.ndarray]:
    """Demonstrate Liouville's theorem: phase-space density preservation.

    Evolves a cloud of initial conditions and shows snapshots at
    successive times. The cloud deforms but preserves area.

    Parameters
    ----------
    system : HamiltonianSystem
    q_center, p_center : ndarray, shape (ndof,)
        Center of the initial cloud.
    spread : float
        Standard deviation of the initial Gaussian cloud.
    n_particles : int
        Number of phase-space points in the cloud.
    dt : float
        Integration time step.
    n_snapshots : int
        Number of time snapshots (including t=0).
    steps_per_snapshot : int
        Integration steps between snapshots.
    q_index, p_index : int
        Which DOF to plot.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple of (Figure, ndarray of Axes)
    """
    q_center = np.asarray(q_center, dtype=float)
    p_center = np.asarray(p_center, dtype=float)
    ndof = system.ndof

    rng = np.random.default_rng(seed)

    # Initialize cloud
    qs = np.tile(q_center, (n_particles, 1)) + spread * rng.standard_normal(
        (n_particles, ndof)
    )
    ps = np.tile(p_center, (n_particles, 1)) + spread * rng.standard_normal(
        (n_particles, ndof)
    )

    cols = min(n_snapshots, 4)
    rows = (n_snapshots + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.5 * rows))
    axes_flat = np.atleast_1d(axes).ravel()

    for snap in range(n_snapshots):
        ax = axes_flat[snap]
        ax.scatter(
            qs[:, q_index], ps[:, p_index],
            s=0.5, color="steelblue", alpha=0.6,
        )
        t_now = snap * steps_per_snapshot * dt
        ax.set_title(f"$t = {t_now:.1f}$")
        ax.set_xlabel(f"$q_{{{q_index + 1}}}$")
        ax.set_ylabel(f"$p_{{{p_index + 1}}}$")
        ax.set_aspect("equal", adjustable="datalim")

        # Evolve all particles to next snapshot
        if snap < n_snapshots - 1:
            for _ in range(steps_per_snapshot):
                for k in range(n_particles):
                    qs[k], ps[k] = stormer_verlet_step(
                        system, qs[k], ps[k], dt
                    )

    # Hide unused subplot panels
    for idx in range(n_snapshots, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("Phase-Space Density Evolution (Liouville)", fontsize=12)
    fig.tight_layout()
    return fig, axes_flat[:n_snapshots]


def plot_chapter4_summary(
    system: HamiltonianSystem,
    q0: np.ndarray,
    p0: np.ndarray,
    dt: float = 0.01,
    steps: int = 50000,
    q_range: tuple[float, float] = (-3.0, 3.0),
    p_range: tuple[float, float] = (-3.0, 3.0),
    poincare_section_index: int | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Generate a 6-panel summary figure for Chapter 4.

    Panels:
        1. Phase portrait
        2. Energy contours
        3. Time series
        4. Energy drift
        5. Poincaré section (2-DOF only) or second phase portrait
        6. Lyapunov convergence

    Parameters
    ----------
    system : HamiltonianSystem
    q0, p0 : ndarray, shape (ndof,)
    dt : float
    steps : int
    q_range, p_range : tuple of float
        For energy contour grid.
    poincare_section_index : int, optional
        Section coordinate index for Poincaré section.
        Defaults to 1 for ndof >= 2, skipped for ndof == 1.

    Returns
    -------
    tuple of (Figure, ndarray of Axes)
    """
    from ..hamiltonian.integrators import integrate_hamiltonian
    from ..hamiltonian.poincare import compute_poincare_section

    q0 = np.asarray(q0, dtype=float)
    p0 = np.asarray(p0, dtype=float)
    ndof = system.ndof

    q_traj, p_traj = integrate_hamiltonian(system, q0, p0, dt=dt, steps=steps)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes_flat = axes.ravel()

    # 1. Phase portrait
    plot_phase_space(q_traj, p_traj, ax=axes_flat[0], title="Phase Portrait")

    # 2. Energy contours
    plot_energy_contours(
        system, q_range=q_range, p_range=p_range,
        ax=axes_flat[1], title="Energy Contours",
    )

    # 3. Time series (show first 10% for clarity)
    n_show = max(steps // 10, 1000)
    plot_time_series(
        q_traj[:n_show], p_traj[:n_show], dt=dt,
        ax=axes_flat[2], title="Time Series",
    )

    # 4. Energy drift
    plot_energy_drift(system, q_traj, p_traj, dt=dt, ax=axes_flat[3])

    # 5. Poincaré section or second phase plot
    if ndof >= 2:
        sec_idx = poincare_section_index if poincare_section_index is not None else 1
        section = compute_poincare_section(q_traj, p_traj, section_index=sec_idx)
        if len(section) > 0:
            plot_poincare_section(section, ax=axes_flat[4])
        else:
            axes_flat[4].text(
                0.5, 0.5, "No crossings", transform=axes_flat[4].transAxes,
                ha="center", va="center",
            )
            axes_flat[4].set_title("Poincaré Section")
    else:
        # For 1-DOF, show energy contours with trajectory overlay
        plot_energy_contours(
            system, q_range=q_range, p_range=p_range,
            ax=axes_flat[4], title="Trajectory on Energy Surface",
            cmap=None, colors="lightgray",
        )
        axes_flat[4].plot(
            q_traj[:, 0], p_traj[:, 0],
            linewidth=0.5, color="crimson",
        )

    # 6. Lyapunov convergence
    plot_lyapunov_convergence(
        system, q0, p0, dt=dt,
        steps=min(steps, 20000), transient=500,
        ax=axes_flat[5],
    )

    fig.suptitle("Chapter 4: Hamiltonian Dynamics Summary", fontsize=14, y=1.01)
    fig.tight_layout()
    return fig, axes_flat
