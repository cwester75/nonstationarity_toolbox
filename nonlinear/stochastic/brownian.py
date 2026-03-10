"""Brownian motion (Wiener process) generator for stochastic baseline comparison."""

import numpy as np


class BrownianMotion:
    """Generate continuous-time Brownian motion paths.

    Parameters
    ----------
    sigma : float
        Volatility (diffusion coefficient, default 1.0).
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(self, sigma=1.0, seed=None):
        self.sigma = sigma
        self.rng = np.random.default_rng(seed)

    def generate(self, n, dt=0.01):
        """Return a Brownian path of *n* points with time step *dt*.

        Parameters
        ----------
        n : int
            Number of points.
        dt : float
            Time step (default 0.01).

        Returns
        -------
        np.ndarray
            Brownian motion path of length *n*.
        """
        steps = self.rng.normal(0, self.sigma * np.sqrt(dt), size=n)
        steps[0] = 0.0
        return np.cumsum(steps)
