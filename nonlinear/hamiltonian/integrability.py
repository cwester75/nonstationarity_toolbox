"""Integrability checks for Hamiltonian systems.

An integrable Hamiltonian system with N degrees of freedom has N
independent conserved quantities (integrals of motion) in involution.
Energy conservation is the most fundamental invariant.
"""

from __future__ import annotations

import numpy as np
from typing import Callable

from .system import HamiltonianSystem


def check_energy_conservation(
    system: HamiltonianSystem,
    q_traj: np.ndarray,
    p_traj: np.ndarray,
    tolerance: float = 1e-6,
) -> tuple[bool, float]:
    """Verify energy conservation along a trajectory.

    Parameters
    ----------
    system : HamiltonianSystem
    q_traj, p_traj : ndarray, shape (steps, ndof)
    tolerance : float
        Maximum allowed relative energy variation.

    Returns
    -------
    tuple of (bool, float)
        (conserved, max_relative_variation). ``conserved`` is True if
        the relative energy variation stays below the tolerance.
    """
    energies = np.array([
        system.energy(q_traj[i], p_traj[i])
        for i in range(len(q_traj))
    ])
    e0 = energies[0]
    if abs(e0) < 1e-15:
        variation = np.max(np.abs(energies - e0))
    else:
        variation = np.max(np.abs((energies - e0) / e0))
    return variation < tolerance, float(variation)


def check_integrability(
    candidate_integrals: list[Callable],
    q_traj: np.ndarray,
    p_traj: np.ndarray,
    tolerance: float = 1e-6,
) -> list[tuple[Callable, float]]:
    """Check which candidate functions are conserved along a trajectory.

    Parameters
    ----------
    candidate_integrals : list of callable
        Each callable has signature I(q, p) -> float.
    q_traj, p_traj : ndarray, shape (steps, ndof)
    tolerance : float
        Maximum allowed variance for a quantity to be considered conserved.

    Returns
    -------
    list of (callable, float)
        Pairs of (integral_function, variance) for candidates whose
        variance along the trajectory is below the tolerance.
    """
    conserved = []
    for I_func in candidate_integrals:
        values = np.array([
            I_func(q_traj[i], p_traj[i])
            for i in range(len(q_traj))
        ])
        var = np.var(values)
        if var < tolerance:
            conserved.append((I_func, float(var)))
    return conserved
