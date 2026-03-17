"""Attractor classification for 2-D dissipative flows.

Classifies the long-term behavior of a trajectory as:
    - fixed point
    - limit cycle
    - strange attractor
    - unbounded (divergent)
"""

from __future__ import annotations

import numpy as np

from ..simulation.integrator import simulate
from .limit_cycles import detect_limit_cycle
from .lyapunov_flow import flow_lyapunov_exponent


def classify_attractor(
    system,
    x0: float,
    y0: float,
    dt: float = 0.01,
    steps: int = 50000,
    transient: int = 5000,
    lyapunov_steps: int = 20000,
    divergence_threshold: float = 1e6,
    fixed_point_threshold: float = 1e-4,
    lyapunov_chaos_threshold: float = 0.01,
) -> dict:
    """Classify the attractor reached from given initial conditions.

    Parameters
    ----------
    system : AutonomousSystem2D
    x0, y0 : float
        Initial conditions.
    dt : float
        Integration time step.
    steps : int
        Total integration steps.
    transient : int
        Steps to discard before analysis.
    lyapunov_steps : int
        Steps for Lyapunov exponent estimation.
    divergence_threshold : float
        Max coordinate magnitude before declaring unbounded.
    fixed_point_threshold : float
        Max variation in tail to declare fixed point convergence.
    lyapunov_chaos_threshold : float
        Positive Lyapunov exponent threshold for chaos.

    Returns
    -------
    dict
        Keys:
            ``'type'`` – one of ``'fixed_point'``, ``'limit_cycle'``,
                         ``'strange_attractor'``, ``'unbounded'``
            ``'lyapunov'`` – estimated maximal Lyapunov exponent
            ``'trajectory'`` – the full trajectory array, shape (steps, 2)
    """
    traj = simulate(system, x0, y0, steps=steps, dt=dt)

    # Check for divergence
    if np.any(np.abs(traj) > divergence_threshold):
        return {
            "type": "unbounded",
            "lyapunov": float("inf"),
            "trajectory": traj,
        }

    tail = traj[transient:]

    # Check for fixed-point convergence
    tail_var = np.max(np.ptp(tail, axis=0))
    if tail_var < fixed_point_threshold:
        return {
            "type": "fixed_point",
            "lyapunov": float("-inf"),
            "trajectory": traj,
        }

    # Check for limit cycle
    is_cycle = detect_limit_cycle(
        tail, section_coord=0, section_value=np.mean(tail[:, 0]),
    )

    # Compute Lyapunov exponent
    lyap = flow_lyapunov_exponent(
        system, x0, y0, dt=dt,
        steps=lyapunov_steps, transient=transient,
    )

    if lyap > lyapunov_chaos_threshold and not is_cycle:
        attractor_type = "strange_attractor"
    elif is_cycle or lyap <= lyapunov_chaos_threshold:
        attractor_type = "limit_cycle"
    else:
        attractor_type = "limit_cycle"

    return {
        "type": attractor_type,
        "lyapunov": lyap,
        "trajectory": traj,
    }


def phase_space_contraction(
    system,
    x: float,
    y: float,
    eps: float = 1e-6,
) -> float:
    """Compute the local phase-space contraction rate (divergence of f).

    For a dissipative system, div(f) < 0 implies volumes contract.

    Parameters
    ----------
    system : AutonomousSystem2D
    x, y : float
        Point at which to evaluate.
    eps : float
        Finite-difference step.

    Returns
    -------
    float
        div(f) = df1/dx + df2/dy
    """
    f_xp = system.vector_field(x + eps, y)
    f_xm = system.vector_field(x - eps, y)
    f_yp = system.vector_field(x, y + eps)
    f_ym = system.vector_field(x, y - eps)

    df1_dx = (f_xp[0] - f_xm[0]) / (2 * eps)
    df2_dy = (f_yp[1] - f_ym[1]) / (2 * eps)

    return float(df1_dx + df2_dy)
