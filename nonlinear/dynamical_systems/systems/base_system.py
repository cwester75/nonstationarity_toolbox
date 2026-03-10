import numpy as np


class AutonomousSystem2D:
    """Base class for autonomous two-dimensional dynamical systems.

    Represents a system of the form:
        dx/dt = f(x, y)
        dy/dt = g(x, y)

    Parameters
    ----------
    f : callable
        Right-hand side for dx/dt.  Signature: f(x, y, params) -> float.
    g : callable
        Right-hand side for dy/dt.  Signature: g(x, y, params) -> float.
    parameters : dict
        Dictionary of system parameters passed to f and g.
    """

    def __init__(self, f, g, parameters=None):
        self.f = f
        self.g = g
        self.params = parameters if parameters is not None else {}

    def vector_field(self, x, y):
        """Evaluate the vector field at (x, y).

        Returns
        -------
        tuple of float
            (dx/dt, dy/dt)
        """
        dx = self.f(x, y, self.params)
        dy = self.g(x, y, self.params)
        return dx, dy

    def phase_portrait(self, x_range, y_range, nx=20, ny=20):
        """Generate vector field data on a grid for visualisation.

        Parameters
        ----------
        x_range : tuple
            (x_min, x_max)
        y_range : tuple
            (y_min, y_max)
        nx, ny : int
            Number of grid points along each axis.

        Returns
        -------
        tuple of ndarray
            (X, Y, DX, DY) meshgrid arrays suitable for quiver plots.
        """
        xs = np.linspace(x_range[0], x_range[1], nx)
        ys = np.linspace(y_range[0], y_range[1], ny)
        X, Y = np.meshgrid(xs, ys)
        DX = np.zeros_like(X)
        DY = np.zeros_like(Y)
        for i in range(ny):
            for j in range(nx):
                DX[i, j], DY[i, j] = self.vector_field(X[i, j], Y[i, j])
        return X, Y, DX, DY
