import numpy as np
from nonlinear.core.map2d import Map2D


class HenonMap(Map2D):
    """Henon map: x_{n+1} = 1 - a*x_n^2 + y_n, y_{n+1} = b*x_n."""

    def __init__(self, a=1.4, b=0.3):
        self.a = a
        self.b = b

    def step(self, state):
        x, y = state
        x_new = 1 - self.a * x ** 2 + y
        y_new = self.b * x
        return (x_new, y_new)

    def jacobian(self, state):
        x, _ = state
        return np.array([
            [-2 * self.a * x, 1.0],
            [self.b, 0.0]
        ])

    def fixed_points(self):
        """Compute fixed points analytically."""
        # x = 1 - a*x^2 + b*x => a*x^2 + (1-b)*x - 1 = 0
        a, b = self.a, self.b
        disc = (1 - b) ** 2 + 4 * a
        if disc < 0:
            return []
        sq = np.sqrt(disc)
        x1 = (-(1 - b) + sq) / (2 * a)
        x2 = (-(1 - b) - sq) / (2 * a)
        return [(x1, b * x1), (x2, b * x2)]
