import numpy as np


def rk4_step(system, x, y, dt):
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
    k1 = system.vector_field(x, y)

    k2 = system.vector_field(
        x + dt * k1[0] / 2,
        y + dt * k1[1] / 2,
    )

    k3 = system.vector_field(
        x + dt * k2[0] / 2,
        y + dt * k2[1] / 2,
    )

    k4 = system.vector_field(
        x + dt * k3[0],
        y + dt * k3[1],
    )

    x_new = x + dt * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6
    y_new = y + dt * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6

    return x_new, y_new


class RK4Integrator:
    """Fourth-order Runge-Kutta integrator for AutonomousSystem2D."""

    def step(self, system, x, y, dt):
        """Advance one RK4 step (class-based interface).

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
        return rk4_step(system, x, y, dt)


def simulate(system, x0, y0, steps=5000, dt=0.01):
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
    ndarray, shape (steps, 2)
        Trajectory array with columns [x, y].  Row 0 is the initial
        condition; row ``steps-1`` is the state after ``steps-1`` steps.
    """
    x, y = float(x0), float(y0)
    traj = np.zeros((steps, 2))

    for i in range(steps):
        traj[i] = [x, y]
        x, y = rk4_step(system, x, y, dt)

    return traj
