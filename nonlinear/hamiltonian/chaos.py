"""Chaos detection for Hamiltonian systems.

Provides:
    - Lyapunov exponent estimation via tangent-space integration
    - Poincaré section geometry classification
"""

from __future__ import annotations

import numpy as np

from .system import HamiltonianSystem
from .integrators import stormer_verlet_step


def hamiltonian_lyapunov(
    system: HamiltonianSystem,
    q0: np.ndarray,
    p0: np.ndarray,
    dt: float = 0.01,
    steps: int = 50000,
    transient: int = 1000,
) -> float:
    """Estimate the maximal Lyapunov exponent of a Hamiltonian trajectory.

    Uses the standard method of integrating two nearby trajectories,
    periodically renormalizing the separation vector.

    Parameters
    ----------
    system : HamiltonianSystem
    q0, p0 : ndarray, shape (ndof,)
        Initial conditions.
    dt : float
        Integration time step.
    steps : int
        Total integration steps (after transient).
    transient : int
        Number of initial steps to discard.

    Returns
    -------
    float
        Estimated maximal Lyapunov exponent.
    """
    q0 = np.asarray(q0, dtype=float)
    p0 = np.asarray(p0, dtype=float)
    ndof = system.ndof

    # Discard transient
    q, p = q0.copy(), p0.copy()
    for _ in range(transient):
        q, p = stormer_verlet_step(system, q, p, dt)

    # Perturbed trajectory
    delta = 1e-9
    dq = np.zeros(ndof)
    dp = np.zeros(ndof)
    dq[0] = delta
    d0 = np.sqrt(np.sum(dq**2) + np.sum(dp**2))

    q2 = q + dq
    p2 = p + dp

    lyap_sum = 0.0
    renorm_interval = 10
    n_renorm = 0

    for i in range(steps):
        q, p = stormer_verlet_step(system, q, p, dt)
        q2, p2 = stormer_verlet_step(system, q2, p2, dt)

        if (i + 1) % renorm_interval == 0:
            sep_q = q2 - q
            sep_p = p2 - p
            dist = np.sqrt(np.sum(sep_q**2) + np.sum(sep_p**2))

            if dist > 0:
                lyap_sum += np.log(dist / d0)
                n_renorm += 1
                factor = d0 / dist
                q2 = q + sep_q * factor
                p2 = p + sep_p * factor

    if n_renorm == 0:
        return 0.0
    return lyap_sum / (n_renorm * renorm_interval * dt)


def classify_poincare_dynamics(
    section_points: np.ndarray,
    curve_tolerance: float = 0.05,
) -> str:
    """Classify dynamics from Poincaré section geometry.

    Applies a simple heuristic based on the spread of section points.

    Parameters
    ----------
    section_points : ndarray, shape (n, d)
        Points on the Poincaré section.
    curve_tolerance : float
        Fraction of the bounding box below which points are considered
        to lie on a curve (regular motion).

    Returns
    -------
    str
        One of 'regular', 'chaotic', or 'insufficient_data'.
    """
    if len(section_points) < 5:
        return "insufficient_data"

    # Compute the spread relative to the bounding box
    ranges = np.ptp(section_points, axis=0)
    max_range = np.max(ranges)

    if max_range < 1e-12:
        return "regular"

    # For regular motion, points should lie near a 1-D curve.
    # Estimate dimensionality via the ratio of singular values.
    centered = section_points - section_points.mean(axis=0)
    _, s, _ = np.linalg.svd(centered, full_matrices=False)
    s = s / s[0]

    # If the second singular value is small, points form a curve
    if len(s) >= 2 and s[1] < curve_tolerance:
        return "regular"

    return "chaotic"
