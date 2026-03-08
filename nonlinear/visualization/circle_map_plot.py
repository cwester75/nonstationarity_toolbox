import numpy as np
import matplotlib.pyplot as plt


def plot_arnold_tongue(locked_points, ax=None):
    """Plot Arnold tongue mode-locked regions.

    Args:
        locked_points: list of (omega, K) tuples from arnold_tongue().
        ax: optional matplotlib axes.
    """
    if ax is None:
        fig, ax = plt.subplots()

    if locked_points:
        omegas, Ks = zip(*locked_points)
        ax.plot(omegas, Ks, ',', markersize=0.5, color='black')

    ax.set_xlabel(r'$\omega$')
    ax.set_ylabel('K')
    ax.set_title('Arnold Tongues')
    return ax


def plot_devils_staircase(omega_values, rotation_numbers, ax=None):
    """Plot the devil's staircase: rotation number vs omega.

    Args:
        omega_values: array of omega parameter values.
        rotation_numbers: array of rotation numbers from rotation_number_scan.
        ax: optional matplotlib axes.
    """
    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(omega_values, rotation_numbers, '-', linewidth=0.5, color='black')
    ax.set_xlabel(r'$\omega$')
    ax.set_ylabel(r'$\rho$')
    ax.set_title("Devil's Staircase")
    ax.grid(True, alpha=0.3)
    return ax


def plot_circle_map_orbit(traj, ax=None):
    """Plot a circle map trajectory on the unit circle.

    Args:
        traj: trajectory of theta values from iterate_circle_map.
        ax: optional matplotlib axes.
    """
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    thetas = 2 * np.pi * np.array(traj)
    radii = np.ones_like(thetas)
    ax.plot(thetas, radii, '.', markersize=2)
    ax.set_title('Circle Map Orbit')
    return ax
