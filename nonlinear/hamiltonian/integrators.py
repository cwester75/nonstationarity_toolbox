"""Symplectic integrators for Hamiltonian systems.

Symplectic integrators preserve the symplectic structure of Hamiltonian
phase space, providing superior long-term energy conservation compared
to generic integrators like RK4.

Implements:
    - Symplectic Euler (1st order)
    - Störmer-Verlet / Leapfrog (2nd order)
"""

from __future__ import annotations

import numpy as np

from .system import HamiltonianSystem


def symplectic_euler_step(
    system: HamiltonianSystem,
    q: np.ndarray,
    p: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance one step using the symplectic Euler method.

    Parameters
    ----------
    system : HamiltonianSystem
    q, p : ndarray, shape (ndof,)
    dt : float

    Returns
    -------
    tuple of ndarray
        (q_new, p_new)
    """
    p_new = p + dt * system.dp_dt(q, p)
    q_new = q + dt * system.dq_dt(q, p_new)
    return q_new, p_new


def stormer_verlet_step(
    system: HamiltonianSystem,
    q: np.ndarray,
    p: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance one step using the Störmer-Verlet (leapfrog) method.

    This is a 2nd-order symplectic integrator:
        p_half = p + (dt/2) * dp/dt(q, p)
        q_new  = q + dt * dq/dt(q, p_half)
        p_new  = p_half + (dt/2) * dp/dt(q_new, p_half)

    Parameters
    ----------
    system : HamiltonianSystem
    q, p : ndarray, shape (ndof,)
    dt : float

    Returns
    -------
    tuple of ndarray
        (q_new, p_new)
    """
    p_half = p + 0.5 * dt * system.dp_dt(q, p)
    q_new = q + dt * system.dq_dt(q, p_half)
    p_new = p_half + 0.5 * dt * system.dp_dt(q_new, p_half)
    return q_new, p_new


# Alias: leapfrog is the same as Störmer-Verlet
leapfrog_step = stormer_verlet_step


def integrate_hamiltonian(
    system: HamiltonianSystem,
    q0: np.ndarray,
    p0: np.ndarray,
    dt: float = 0.01,
    steps: int = 10000,
    method: str = "verlet",
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate a Hamiltonian system over many time steps.

    Parameters
    ----------
    system : HamiltonianSystem
    q0, p0 : ndarray, shape (ndof,)
        Initial conditions.
    dt : float
        Time step.
    steps : int
        Number of integration steps.
    method : str
        Integration method: 'verlet' (default), 'leapfrog', or 'euler'.

    Returns
    -------
    tuple of ndarray
        (q_traj, p_traj) each of shape (steps, ndof).
        Row 0 is the initial condition.
    """
    q0 = np.asarray(q0, dtype=float)
    p0 = np.asarray(p0, dtype=float)
    ndof = system.ndof

    q_traj = np.zeros((steps, ndof))
    p_traj = np.zeros((steps, ndof))

    if method in ("verlet", "leapfrog"):
        step_fn = stormer_verlet_step
    elif method == "euler":
        step_fn = symplectic_euler_step
    else:
        raise ValueError(f"Unknown method '{method}'; use 'verlet' or 'euler'")

    q, p = q0.copy(), p0.copy()
    for i in range(steps):
        q_traj[i] = q
        p_traj[i] = p
        q, p = step_fn(system, q, p, dt)

    return q_traj, p_traj
