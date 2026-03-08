import numpy as np


class Map2D:
    """Base class for two-dimensional iterated maps."""

    def step(self, state):
        """Compute one iteration. state is (x, y). Returns (x_new, y_new)."""
        raise NotImplementedError

    def jacobian(self, state):
        """Return the 2x2 Jacobian matrix at the given state."""
        raise NotImplementedError

    def iterate(self, state, n):
        """Iterate the map n times, returning full trajectory."""
        traj = [np.array(state, dtype=float)]
        s = np.array(state, dtype=float)
        for _ in range(n):
            s = np.array(self.step(s), dtype=float)
            traj.append(s)
        return np.array(traj)
