import numpy as np


class LogisticMap:
    def __init__(self, r):
        self.r = r

    def f(self, x):
        return self.r * x * (1 - x)

    def df(self, x):
        return self.r * (1 - 2 * x)


class TentMap:
    def __init__(self, mu=2):
        self.mu = mu

    def f(self, x):
        if x < 0.5:
            return self.mu * x
        return self.mu * (1 - x)

    def df(self, x):
        return self.mu if x < 0.5 else -self.mu
