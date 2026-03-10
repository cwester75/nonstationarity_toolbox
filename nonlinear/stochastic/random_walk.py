"""Discrete random-walk generator for stochastic baseline comparison."""

import numpy as np


class RandomWalk:
    """Generate discrete random walks.

    Parameters
    ----------
    sigma : float
        Standard deviation of each step (default 1.0).
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(self, sigma=1.0, seed=None):
        self.sigma = sigma
        self.rng = np.random.default_rng(seed)

    def generate(self, n):
        """Return a random walk of length *n* starting at 0."""
        steps = self.rng.normal(0, self.sigma, size=n)
        steps[0] = 0.0
        return np.cumsum(steps)
