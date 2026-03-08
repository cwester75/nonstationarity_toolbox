import numpy as np
from nonlinear.core.map2d import Map2D


class BakerMap(Map2D):
    """Baker's map: stretching and folding transformation on the unit square."""

    def step(self, state):
        x, y = state
        if x < 0.5:
            return (2 * x, y / 2)
        else:
            return (2 * x - 1, (y + 1) / 2)

    def jacobian(self, state):
        return np.array([
            [2.0, 0.0],
            [0.0, 0.5]
        ])


class GeneralizedBakerMap(Map2D):
    """Generalized baker's map with parameter alpha controlling the partition."""

    def __init__(self, alpha=0.5):
        self.alpha = alpha

    def step(self, state):
        x, y = state
        a = self.alpha
        if x < a:
            return (x / a, a * y)
        else:
            return ((x - a) / (1 - a), (1 - a) * y + a)

    def jacobian(self, state):
        x, _ = state
        a = self.alpha
        if x < a:
            return np.array([
                [1.0 / a, 0.0],
                [0.0, a]
            ])
        else:
            return np.array([
                [1.0 / (1 - a), 0.0],
                [0.0, 1 - a]
            ])
