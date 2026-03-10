import numpy as np
import matplotlib.pyplot as plt


# ── colour map for stability types ──────────────────────────────────

_STABILITY_COLORS = {
    "stable node": "blue",
    "unstable node": "red",
    "saddle": "orange",
    "stable spiral": "dodgerblue",
    "unstable spiral": "tomato",
    "center": "green",
    "undetermined": "grey",
    "non-isolated": "grey",
}

_STABILITY_MARKERS = {
    "stable node": "o",
    "unstable node": "o",
    "saddle": "X",
    "stable spiral": "o",
    "unstable spiral": "o",
    "center": "D",
    "undetermined": "s",
    "non-isolated": "s",
}


# ── vector field ────────────────────────────────────────────────────

def plot_vector_field(system, x_range, y_range, nx=20, ny=20,
                      ax=None, normalize=True, **kwargs):
    """Plot the vector field of a 2-D autonomous system as a quiver plot.

    Parameters
    ----------
    system : AutonomousSystem2D
    x_range, y_range : tuple
        (min, max) for each axis.
    nx, ny : int
        Grid resolution.
    ax : matplotlib Axes, optional
    normalize : bool
        If True, arrows have uniform length (direction-only field).
    **kwargs
        Forwarded to ``ax.quiver``.

    Returns
    -------
    matplotlib Axes
    """
    X, Y, DX, DY = system.phase_portrait(x_range, y_range, nx, ny)

    if normalize:
        mag = np.sqrt(DX ** 2 + DY ** 2)
        mag[mag == 0] = 1.0
        DX = DX / mag
        DY = DY / mag

    if ax is None:
        fig, ax = plt.subplots()

    kw = dict(angles='xy', pivot='mid', alpha=0.6)
    kw.update(kwargs)
    ax.quiver(X, Y, DX, DY, **kw)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Vector Field')
    return ax


def plot_streamlines(system, x_range, y_range, nx=100, ny=100,
                     ax=None, density=1.5, **kwargs):
    """Plot the vector field as a streamline diagram.

    Parameters
    ----------
    system : AutonomousSystem2D
    x_range, y_range : tuple
    nx, ny : int
        Grid resolution for the underlying field.
    ax : matplotlib Axes, optional
    density : float
        Streamline density (forwarded to ``ax.streamplot``).
    **kwargs
        Forwarded to ``ax.streamplot``.

    Returns
    -------
    matplotlib Axes
    """
    X, Y, DX, DY = system.phase_portrait(x_range, y_range, nx, ny)
    speed = np.sqrt(DX ** 2 + DY ** 2)

    if ax is None:
        fig, ax = plt.subplots()

    kw = dict(color=speed, cmap='coolwarm', density=density, linewidth=0.7)
    kw.update(kwargs)
    ax.streamplot(X, Y, DX, DY, **kw)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Streamlines')
    return ax


# ── trajectories ────────────────────────────────────────────────────

def plot_trajectory(trajectory, ax=None, **kwargs):
    """Plot a single trajectory in the phase plane.

    Parameters
    ----------
    trajectory : ndarray, shape (N, 2)
        Pre-computed trajectory (e.g. from ``simulate``).
    ax : matplotlib Axes, optional
    **kwargs
        Forwarded to ``ax.plot``.

    Returns
    -------
    matplotlib Axes
    """
    if ax is None:
        fig, ax = plt.subplots()

    kw = dict(linewidth=0.8)
    kw.update(kwargs)
    ax.plot(trajectory[:, 0], trajectory[:, 1], **kw)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Phase Trajectory')
    return ax


def plot_trajectories(trajectories, ax=None, **kwargs):
    """Overlay multiple trajectories on one phase plane.

    Parameters
    ----------
    trajectories : list of ndarray
        Each element has shape (N, 2).
    ax : matplotlib Axes, optional
    **kwargs
        Forwarded to ``ax.plot`` for every trajectory.

    Returns
    -------
    matplotlib Axes
    """
    if ax is None:
        fig, ax = plt.subplots()

    kw = dict(linewidth=0.6, alpha=0.7)
    kw.update(kwargs)
    for traj in trajectories:
        ax.plot(traj[:, 0], traj[:, 1], **kw)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Phase Trajectories')
    return ax


# ── fixed points ────────────────────────────────────────────────────

def plot_fixed_points(fixed_points, classifications=None, ax=None, **kwargs):
    """Plot fixed points on the phase plane, colour-coded by stability.

    Parameters
    ----------
    fixed_points : list of array-like
        Each element is (x, y).
    classifications : list of str, optional
        Stability labels from ``classify_fixed_point``.  If *None*,
        all points are drawn in black.
    ax : matplotlib Axes, optional
    **kwargs
        Forwarded to ``ax.scatter``.

    Returns
    -------
    matplotlib Axes
    """
    if ax is None:
        fig, ax = plt.subplots()

    if classifications is None:
        classifications = ["undetermined"] * len(fixed_points)

    for fp, cls in zip(fixed_points, classifications):
        color = _STABILITY_COLORS.get(cls, "grey")
        marker = _STABILITY_MARKERS.get(cls, "o")
        kw = dict(s=80, zorder=5, edgecolors='k', linewidths=0.8)
        kw.update(kwargs)
        ax.scatter(fp[0], fp[1], c=color, marker=marker, label=cls, **kw)

    # De-duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=8)
    return ax


# ── Poincaré section ───────────────────────────────────────────────

def plot_poincare_section(trajectory, section_coord=0, section_value=0.0,
                          ax=None, **kwargs):
    """Plot the Poincaré return map from section crossings.

    Collects upward crossings of ``trajectory[:, section_coord]``
    through *section_value* and plots the other coordinate of
    successive crossings against one another.

    Parameters
    ----------
    trajectory : ndarray, shape (N, 2)
    section_coord : int
        0 for x-section, 1 for y-section.
    section_value : float
    ax : matplotlib Axes, optional
    **kwargs
        Forwarded to ``ax.plot``.

    Returns
    -------
    matplotlib Axes
    """
    other = 1 - section_coord
    crossings = []

    for i in range(1, len(trajectory)):
        prev = trajectory[i - 1, section_coord] - section_value
        curr = trajectory[i, section_coord] - section_value
        if prev < 0 and curr >= 0:
            crossings.append(trajectory[i, other])

    crossings = np.array(crossings)

    if ax is None:
        fig, ax = plt.subplots()

    if len(crossings) > 1:
        kw = dict(marker='.', linestyle='none', markersize=3)
        kw.update(kwargs)
        ax.plot(crossings[:-1], crossings[1:], **kw)

    coord_name = 'y' if section_coord == 0 else 'x'
    ax.set_xlabel(f'{coord_name}_n')
    ax.set_ylabel(f'{coord_name}_{{n+1}}')
    ax.set_title('Poincaré Return Map')
    return ax


# ── composite phase portrait ───────────────────────────────────────

def plot_phase_portrait_composite(system, trajectory=None,
                                  fixed_points=None,
                                  classifications=None,
                                  x_range=None, y_range=None,
                                  ax=None, show_field=True):
    """All-in-one phase portrait: vector field + trajectory + fixed points.

    Parameters
    ----------
    system : AutonomousSystem2D
    trajectory : ndarray, optional
        Shape (N, 2).
    fixed_points : list, optional
    classifications : list of str, optional
    x_range, y_range : tuple, optional
        If *None* and a trajectory is given, ranges are inferred from data.
    ax : matplotlib Axes, optional
    show_field : bool
        Whether to overlay the normalised quiver field.

    Returns
    -------
    matplotlib Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Infer ranges from trajectory if not given
    if trajectory is not None and x_range is None:
        margin = 0.1
        xmin, xmax = trajectory[:, 0].min(), trajectory[:, 0].max()
        ymin, ymax = trajectory[:, 1].min(), trajectory[:, 1].max()
        dx = (xmax - xmin) * margin or 0.5
        dy = (ymax - ymin) * margin or 0.5
        x_range = (xmin - dx, xmax + dx)
        y_range = (ymin - dy, ymax + dy)

    if show_field and x_range is not None and y_range is not None:
        plot_vector_field(system, x_range, y_range, ax=ax)

    if trajectory is not None:
        plot_trajectory(trajectory, ax=ax)

    if fixed_points is not None:
        plot_fixed_points(fixed_points, classifications, ax=ax)

    ax.set_title('Phase Portrait')
    return ax
