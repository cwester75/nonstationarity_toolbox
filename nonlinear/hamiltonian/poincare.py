"""Poincaré surface of section for Hamiltonian systems.

Reduces continuous-time dynamics to a discrete map by recording
phase-space coordinates whenever the trajectory crosses a
user-defined surface.
"""

from __future__ import annotations

import numpy as np
from typing import Callable


def interpolate_crossing(
    state1: np.ndarray,
    state2: np.ndarray,
    val1: float,
    val2: float,
) -> np.ndarray:
    """Linearly interpolate the crossing point between two trajectory states.

    Parameters
    ----------
    state1, state2 : ndarray
        Full phase-space states at consecutive time steps.
    val1, val2 : float
        Values of the section function at the two states.

    Returns
    -------
    ndarray
        Interpolated state at the zero crossing.
    """
    t = val1 / (val1 - val2)
    return state1 + t * (state2 - state1)


def compute_poincare_section(
    q_traj: np.ndarray,
    p_traj: np.ndarray,
    section_index: int = 1,
    section_value: float = 0.0,
    direction: int = 1,
    interpolate: bool = True,
) -> np.ndarray:
    """Compute the Poincaré surface of section from a trajectory.

    Detects crossings of the plane ``q[section_index] = section_value``
    in the specified direction and returns the remaining phase-space
    coordinates at each crossing.

    Parameters
    ----------
    q_traj : ndarray, shape (steps, ndof)
        Position trajectory.
    p_traj : ndarray, shape (steps, ndof)
        Momentum trajectory.
    section_index : int
        Which q coordinate defines the section plane.
    section_value : float
        Value of q[section_index] at the section plane.
    direction : int
        +1 for crossings from below, -1 for crossings from above.
    interpolate : bool
        Whether to linearly interpolate the crossing point.

    Returns
    -------
    ndarray, shape (n_crossings, 2*ndof - 1)
        Each row contains the phase-space coordinates at the crossing,
        excluding q[section_index] (which is approximately section_value).
        For a 2-DOF system the columns are [q_other, p_section, p_other].
    """
    steps, ndof = q_traj.shape
    crossings = []

    for i in range(steps - 1):
        v1 = q_traj[i, section_index] - section_value
        v2 = q_traj[i + 1, section_index] - section_value

        if direction == 1 and v1 < 0 and v2 >= 0:
            crossed = True
        elif direction == -1 and v1 > 0 and v2 <= 0:
            crossed = True
        else:
            crossed = False

        if crossed:
            # Build full state vectors
            state1 = np.concatenate([q_traj[i], p_traj[i]])
            state2 = np.concatenate([q_traj[i + 1], p_traj[i + 1]])

            if interpolate:
                state = interpolate_crossing(state1, state2, v1, v2)
            else:
                state = state2

            # Remove the section coordinate
            keep = list(range(2 * ndof))
            keep.pop(section_index)
            crossings.append(state[keep])

    if len(crossings) == 0:
        return np.empty((0, 2 * ndof - 1))
    return np.array(crossings)
