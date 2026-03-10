import numpy as np

from ..simulation.integrator import simulate


def detect_homoclinic(system, saddle, n_directions=8, perturbation=1e-3,
                      steps=10000, dt=0.01, tol=1e-2):
    """Detect whether a homoclinic orbit exists near a saddle point.

    Launches trajectories from small perturbations around the saddle
    and checks whether any return close to it.

    Parameters
    ----------
    system : AutonomousSystem2D
    saddle : array-like, shape (2,)
        Coordinates of the saddle point.
    n_directions : int
        Number of perturbation directions to try.
    perturbation : float
        Magnitude of the initial perturbation.
    steps : int
        Number of integration steps per trajectory.
    dt : float
        Integration time step.
    tol : float
        Distance threshold for considering a return.

    Returns
    -------
    bool
        True if a homoclinic-like orbit is detected.
    """
    x0, y0 = float(saddle[0]), float(saddle[1])
    angles = np.linspace(0, 2 * np.pi, n_directions, endpoint=False)

    for angle in angles:
        px = perturbation * np.cos(angle)
        py = perturbation * np.sin(angle)

        traj = simulate(system, x0 + px, y0 + py, steps, dt)

        # Check the last portion of the trajectory for return
        tail = traj[steps // 2:]
        distances = np.sqrt((tail[:, 0] - x0) ** 2 + (tail[:, 1] - y0) ** 2)

        if np.any(distances < tol):
            return True

    return False
