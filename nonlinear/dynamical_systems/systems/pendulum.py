import numpy as np

from .base_system import AutonomousSystem2D


class Pendulum(AutonomousSystem2D):
    """Simple (undamped, undriven) pendulum in the phase plane.

    Equations:
        dx/dt = y
        dy/dt = -sin(x)

    where x is the angle and y is the angular velocity.
    """

    def __init__(self):
        def f(x, y, p):
            return y

        def g(x, y, p):
            return -np.sin(x)

        super().__init__(f, g, {})


class DampedPendulum(AutonomousSystem2D):
    """Damped pendulum in the phase plane.

    Equations:
        dx/dt = y
        dy/dt = -sin(x) - b*y

    Parameters
    ----------
    b : float
        Damping coefficient.
    """

    def __init__(self, b=0.1):
        params = {"b": b}

        def f(x, y, p):
            return y

        def g(x, y, p):
            return -np.sin(x) - p["b"] * y

        super().__init__(f, g, params)
