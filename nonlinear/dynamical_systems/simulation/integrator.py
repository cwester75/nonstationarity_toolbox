import numpy as np


class RK4Integrator:
    """Fourth-order Runge-Kutta integrator for AutonomousSystem2D."""

    def step(self, system, x, y, dt):
        """Advance one RK4 step.

        Parameters
        ----------
        system : AutonomousSystem2D
        x, y : float
            Current state.
        dt : float
            Time step.

        Returns
        -------
        tuple of float
            (x_new, y_new)
        """
        k1x, k1y = system.vector_field(x, y)

        k2x, k2y = system.vector_field(
            x + dt * k1x / 2,
            y + dt * k1y / 2,
        )

        k3x, k3y = system.vector_field(
            x + dt * k2x / 2,
            y + dt * k2y / 2,
        )

        k4x, k4y = system.vector_field(
            x + dt * k3x,
            y + dt * k3y,
        )

        x_new = x + dt * (k1x + 2 * k2x + 2 * k3x + k4x) / 6
        y_new = y + dt * (k1y + 2 * k2y + 2 * k3y + k4y) / 6

        return x_new, y_new


def simulate(system, x0, y0, steps, dt):
    """Simulate a trajectory using RK4 integration.

    Parameters
    ----------
    system : AutonomousSystem2D
    x0, y0 : float
        Initial conditions.
    steps : int
        Number of integration steps.
    dt : float
        Time step.

    Returns
    -------
    ndarray, shape (steps+1, 2)
        Trajectory array with columns [x, y].
    """
    integrator = RK4Integrator()
    traj = np.empty((steps + 1, 2))
    traj[0] = (x0, y0)
    x, y = float(x0), float(y0)
    for i in range(steps):
        x, y = integrator.step(system, x, y, dt)
        traj[i + 1] = (x, y)
    return traj
