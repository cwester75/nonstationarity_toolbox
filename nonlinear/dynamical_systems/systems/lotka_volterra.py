from .base_system import AutonomousSystem2D


class LotkaVolterra(AutonomousSystem2D):
    """Lotka-Volterra predator-prey model.

    Equations:
        dx/dt = alpha*x - beta*x*y
        dy/dt = delta*x*y - gamma*y

    Parameters
    ----------
    alpha : float
        Prey growth rate.
    beta : float
        Predation rate.
    delta : float
        Predator growth efficiency.
    gamma : float
        Predator death rate.
    """

    def __init__(self, alpha=1.5, beta=1.0, delta=1.0, gamma=3.0):
        params = {"alpha": alpha, "beta": beta, "delta": delta, "gamma": gamma}

        def f(x, y, p):
            return p["alpha"] * x - p["beta"] * x * y

        def g(x, y, p):
            return p["delta"] * x * y - p["gamma"] * y

        super().__init__(f, g, params)
