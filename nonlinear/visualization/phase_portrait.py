import numpy as np
import matplotlib.pyplot as plt


def plot_orbit_2d(map2d, state0, n=1000, transient=200, ax=None):
    """Plot the orbit of a 2-D map."""
    traj = map2d.iterate(state0, n + transient)
    traj = traj[transient:]

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(traj[:, 0], traj[:, 1], ',', markersize=0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Phase Portrait')
    return ax


def plot_attractor(map2d, state0, n=5000, transient=500, ax=None, **kwargs):
    """Plot the attractor of a 2-D map."""
    traj = map2d.iterate(state0, n + transient)
    traj = traj[transient:]

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(traj[:, 0], traj[:, 1], ',', markersize=0.3, **kwargs)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    return ax


def plot_phase_portrait(xs, ys, ax=None):
    """Plot a phase portrait from pre-computed x and y arrays."""
    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(xs, ys, '.', markersize=1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Phase Portrait')
    return ax


def plot_multiple_orbits(map2d, initial_conditions, n=500, ax=None):
    """Plot multiple orbits on the same phase portrait."""
    if ax is None:
        fig, ax = plt.subplots()

    for ic in initial_conditions:
        traj = map2d.iterate(ic, n)
        ax.plot(traj[:, 0], traj[:, 1], '-', linewidth=0.5, alpha=0.7)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    return ax
