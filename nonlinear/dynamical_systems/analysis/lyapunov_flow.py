"""Lyapunov exponent estimation for continuous-time 2-D flows.

Provides:
    - Maximal Lyapunov exponent via twin-trajectory renormalization
    - Full 2-D Lyapunov spectrum via QR decomposition of the
      variational equation
"""

from __future__ import annotations

import numpy as np

from ..simulation.integrator import rk4_step
from .fixed_points import jacobian


def flow_lyapunov_exponent(
    system,
    x0: float,
    y0: float,
    dt: float = 0.01,
    steps: int = 50000,
    transient: int = 1000,
    delta: float = 1e-9,
    renorm_interval: int = 10,
) -> float:
    """Estimate the maximal Lyapunov exponent of a 2-D flow.

    Integrates two nearby trajectories with periodic renormalization.

    Parameters
    ----------
    system : AutonomousSystem2D
    x0, y0 : float
        Initial conditions.
    dt : float
        Integration time step.
    steps : int
        Total steps after transient.
    transient : int
        Steps to discard.
    delta : float
        Initial separation magnitude.
    renorm_interval : int
        Steps between renormalizations.

    Returns
    -------
    float
        Estimated maximal Lyapunov exponent (in units of 1/time).
    """
    x, y = float(x0), float(y0)

    # Discard transient
    for _ in range(transient):
        x, y = rk4_step(system, x, y, dt)

    # Perturbed trajectory
    x2 = x + delta
    y2 = y
    d0 = delta

    lyap_sum = 0.0
    n_renorm = 0

    for i in range(steps):
        x, y = rk4_step(system, x, y, dt)
        x2, y2 = rk4_step(system, x2, y2, dt)

        if (i + 1) % renorm_interval == 0:
            dx = x2 - x
            dy = y2 - y
            dist = np.sqrt(dx**2 + dy**2)

            if dist > 0:
                lyap_sum += np.log(dist / d0)
                n_renorm += 1
                factor = d0 / dist
                x2 = x + dx * factor
                y2 = y + dy * factor

    if n_renorm == 0:
        return 0.0
    return lyap_sum / (n_renorm * renorm_interval * dt)


def flow_lyapunov_spectrum(
    system,
    x0: float,
    y0: float,
    dt: float = 0.01,
    steps: int = 50000,
    transient: int = 1000,
) -> np.ndarray:
    """Compute the full Lyapunov spectrum of a 2-D flow via QR decomposition.

    Integrates the variational equation alongside the trajectory,
    periodically orthonormalizing the tangent vectors.

    Parameters
    ----------
    system : AutonomousSystem2D
    x0, y0 : float
        Initial conditions.
    dt : float
        Integration time step.
    steps : int
        Total steps after transient.
    transient : int
        Steps to discard.

    Returns
    -------
    ndarray, shape (2,)
        Lyapunov exponents [lambda_1, lambda_2], sorted descending.
    """
    x, y = float(x0), float(y0)

    # Discard transient
    for _ in range(transient):
        x, y = rk4_step(system, x, y, dt)

    dim = 2
    Q = np.eye(dim)
    lyap_sums = np.zeros(dim)

    for _ in range(steps):
        J = jacobian(system, x, y)
        x, y = rk4_step(system, x, y, dt)

        # Linearized flow: Phi ≈ I + dt*J (first-order)
        Phi = np.eye(dim) + dt * J
        M = Phi @ Q
        Q, R = np.linalg.qr(M)

        for j in range(dim):
            if abs(R[j, j]) > 0:
                lyap_sums[j] += np.log(abs(R[j, j]))

    exponents = lyap_sums / (steps * dt)
    return np.sort(exponents)[::-1]


def flow_lyapunov_convergence(
    system,
    x0: float,
    y0: float,
    dt: float = 0.01,
    steps: int = 50000,
    transient: int = 1000,
    delta: float = 1e-9,
    renorm_interval: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the running Lyapunov exponent estimate over time.

    Parameters
    ----------
    system : AutonomousSystem2D
    x0, y0 : float
    dt, steps, transient, delta, renorm_interval :
        Same as ``flow_lyapunov_exponent``.

    Returns
    -------
    tuple of ndarray
        (times, estimates) arrays for plotting convergence.
    """
    x, y = float(x0), float(y0)

    for _ in range(transient):
        x, y = rk4_step(system, x, y, dt)

    x2 = x + delta
    y2 = y
    d0 = delta

    lyap_sum = 0.0
    n_renorm = 0
    times = []
    estimates = []

    for i in range(steps):
        x, y = rk4_step(system, x, y, dt)
        x2, y2 = rk4_step(system, x2, y2, dt)

        if (i + 1) % renorm_interval == 0:
            dx = x2 - x
            dy = y2 - y
            dist = np.sqrt(dx**2 + dy**2)

            if dist > 0:
                lyap_sum += np.log(dist / d0)
                n_renorm += 1
                factor = d0 / dist
                x2 = x + dx * factor
                y2 = y + dy * factor

                times.append((i + 1) * dt)
                estimates.append(
                    lyap_sum / (n_renorm * renorm_interval * dt)
                )

    return np.array(times), np.array(estimates)
