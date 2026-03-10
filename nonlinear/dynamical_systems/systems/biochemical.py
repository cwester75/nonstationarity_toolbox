from .base_system import AutonomousSystem2D


class BiochemicalSystem(AutonomousSystem2D):
    """Simple enzyme-kinetics reaction system.

    Equations:
        dx/dt = k1 - k2*x*y
        dy/dt = k3*x - k4*y

    Parameters
    ----------
    k1, k2, k3, k4 : float
        Reaction rate constants.
    """

    def __init__(self, k1=1.0, k2=1.0, k3=1.0, k4=1.0):
        params = {"k1": k1, "k2": k2, "k3": k3, "k4": k4}

        def f(x, y, p):
            return p["k1"] - p["k2"] * x * y

        def g(x, y, p):
            return p["k3"] * x - p["k4"] * y

        super().__init__(f, g, params)
