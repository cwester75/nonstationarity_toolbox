"""Action-angle variable computation for integrable Hamiltonian systems.

For a 1-DOF integrable system, the action variable is:
    I = (1 / 2π) ∮ p dq

The angle variable θ advances linearly in time:
    θ(t) = ω t  (mod 2π)
"""

from __future__ import annotations

import numpy as np


def compute_action_variable(q: np.ndarray, p: np.ndarray) -> float:
    """Compute the action variable from a closed orbit in (q, p) space.

    Uses the trapezoidal rule to approximate the loop integral
    I = (1/2π) ∮ p dq.

    Parameters
    ----------
    q : ndarray, shape (n,)
        Position coordinates along one full orbit.
    p : ndarray, shape (n,)
        Momentum coordinates along one full orbit.

    Returns
    -------
    float
        The action variable I.
    """
    dq = np.diff(q)
    p_mid = 0.5 * (p[:-1] + p[1:])
    integral = np.sum(p_mid * dq)
    return integral / (2 * np.pi)


def compute_angle_variable(omega: float, t: float | np.ndarray) -> float | np.ndarray:
    """Compute the angle variable θ = ωt (mod 2π).

    Parameters
    ----------
    omega : float
        Angular frequency of the motion.
    t : float or ndarray
        Time(s) at which to evaluate the angle.

    Returns
    -------
    float or ndarray
        Angle variable(s) in [0, 2π).
    """
    return np.mod(omega * t, 2 * np.pi)
