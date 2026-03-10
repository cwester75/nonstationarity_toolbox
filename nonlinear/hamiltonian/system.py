"""Core Hamiltonian system representation.

A Hamiltonian system with N degrees of freedom evolves via:
    dq_i/dt =  dH/dp_i
    dp_i/dt = -dH/dq_i

This module provides the base class that wraps a user-supplied Hamiltonian
function H(q, p) and derives the equations of motion via finite differences.
"""

from __future__ import annotations

import numpy as np
from typing import Callable


class HamiltonianSystem:
    """Hamiltonian dynamical system with N degrees of freedom.

    Parameters
    ----------
    H : callable
        Hamiltonian function with signature H(q, p) -> float,
        where q and p are 1-D arrays of length N.
    ndof : int
        Number of degrees of freedom.
    epsilon : float, optional
        Step size for finite-difference gradient computation.
    """

    def __init__(
        self,
        H: Callable[[np.ndarray, np.ndarray], float],
        ndof: int = 1,
        epsilon: float = 1e-7,
    ):
        self.H = H
        self.ndof = ndof
        self.epsilon = epsilon

    def _gradient(self, H_func, q, p, wrt="p"):
        """Compute the gradient of H via central finite differences.

        Parameters
        ----------
        H_func : callable
        q, p : ndarray
        wrt : str
            'p' for dH/dp, 'q' for dH/dq.

        Returns
        -------
        ndarray
        """
        var = p.copy() if wrt == "p" else q.copy()
        grad = np.zeros_like(var)
        eps = self.epsilon
        for i in range(len(var)):
            var_plus = var.copy()
            var_minus = var.copy()
            var_plus[i] += eps
            var_minus[i] -= eps
            if wrt == "p":
                grad[i] = (H_func(q, var_plus) - H_func(q, var_minus)) / (2 * eps)
            else:
                grad[i] = (H_func(var_plus, p) - H_func(var_minus, p)) / (2 * eps)
        return grad

    def dq_dt(self, q: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Compute dq/dt = dH/dp.

        Parameters
        ----------
        q, p : ndarray, shape (ndof,)

        Returns
        -------
        ndarray, shape (ndof,)
        """
        return self._gradient(self.H, q, p, wrt="p")

    def dp_dt(self, q: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Compute dp/dt = -dH/dq.

        Parameters
        ----------
        q, p : ndarray, shape (ndof,)

        Returns
        -------
        ndarray, shape (ndof,)
        """
        return -self._gradient(self.H, q, p, wrt="q")

    def energy(self, q: np.ndarray, p: np.ndarray) -> float:
        """Evaluate the Hamiltonian (total energy) at a phase-space point.

        Parameters
        ----------
        q, p : ndarray, shape (ndof,)

        Returns
        -------
        float
        """
        return float(self.H(q, p))

    def equations_of_motion(self, q: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Return the full state derivative [dq/dt, dp/dt].

        Parameters
        ----------
        q, p : ndarray, shape (ndof,)

        Returns
        -------
        ndarray, shape (2*ndof,)
        """
        return np.concatenate([self.dq_dt(q, p), self.dp_dt(q, p)])
