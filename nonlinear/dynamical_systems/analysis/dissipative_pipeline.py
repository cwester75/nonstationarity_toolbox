"""Chapter 5 analysis pipeline for nonlinear dissipative systems.

Orchestrates the full diagnostic workflow:
    1. Fixed-point finding and stability classification
    2. Trajectory simulation
    3. Limit cycle detection
    4. Lyapunov exponent and spectrum estimation
    5. Attractor classification
    6. Hopf bifurcation detection (optional parameter sweep)
"""

from __future__ import annotations

import numpy as np

from ..simulation.integrator import simulate
from .fixed_points import find_fixed_points, jacobian
from .stability import classify_fixed_point
from .limit_cycles import detect_limit_cycle
from .lyapunov_flow import (
    flow_lyapunov_exponent,
    flow_lyapunov_spectrum,
    flow_lyapunov_convergence,
)
from .attractors import classify_attractor, phase_space_contraction
from .homoclinic import detect_homoclinic


def dissipative_analysis(
    system,
    x0: float,
    y0: float,
    dt: float = 0.01,
    steps: int = 50000,
    transient: int = 5000,
    fixed_point_guesses: list[tuple[float, float]] | None = None,
) -> dict:
    """Run the full Chapter 5 dissipative-system analysis pipeline.

    Parameters
    ----------
    system : AutonomousSystem2D
    x0, y0 : float
        Initial conditions for trajectory simulation.
    dt : float
        Integration time step.
    steps : int
        Total simulation steps.
    transient : int
        Steps to discard for post-transient analysis.
    fixed_point_guesses : list of (float, float), optional
        Starting guesses for fixed-point search.  Defaults to a grid
        around the origin.

    Returns
    -------
    dict
        Comprehensive results with keys:

        ``'equilibria'``
            List of dicts, each with ``'point'``, ``'classification'``,
            ``'eigenvalues'``, ``'divergence'``.
        ``'trajectory'``
            ndarray, shape (steps, 2).
        ``'limit_cycle'``
            bool – whether a limit cycle is detected.
        ``'lyapunov_exponent'``
            float – maximal Lyapunov exponent.
        ``'lyapunov_spectrum'``
            ndarray, shape (2,) – full spectrum [lambda1, lambda2].
        ``'lyapunov_convergence'``
            tuple (times, estimates) arrays.
        ``'attractor'``
            dict from ``classify_attractor``.
        ``'homoclinic'``
            dict mapping saddle index to bool.
    """
    # 1. Fixed points
    if fixed_point_guesses is None:
        fixed_point_guesses = [
            (i, j)
            for i in np.linspace(-2, 2, 5)
            for j in np.linspace(-2, 2, 5)
        ]

    fps = find_fixed_points(system, fixed_point_guesses)

    equilibria = []
    for fp in fps:
        cls = classify_fixed_point(system, fp)
        eigs = np.linalg.eigvals(jacobian(system, fp[0], fp[1]))
        div = phase_space_contraction(system, fp[0], fp[1])
        equilibria.append({
            "point": fp,
            "classification": cls,
            "eigenvalues": eigs,
            "divergence": div,
        })

    # 2. Trajectory simulation
    traj = simulate(system, x0, y0, steps=steps, dt=dt)

    # 3. Limit cycle detection
    tail = traj[transient:]
    is_cycle = detect_limit_cycle(
        tail, section_coord=0,
        section_value=np.mean(tail[:, 0]),
    )

    # 4. Lyapunov exponents
    lyap_max = flow_lyapunov_exponent(
        system, x0, y0, dt=dt,
        steps=min(steps, 30000), transient=transient,
    )
    lyap_spectrum = flow_lyapunov_spectrum(
        system, x0, y0, dt=dt,
        steps=min(steps, 30000), transient=transient,
    )
    lyap_times, lyap_estimates = flow_lyapunov_convergence(
        system, x0, y0, dt=dt,
        steps=min(steps, 30000), transient=transient,
    )

    # 5. Attractor classification
    attractor = classify_attractor(
        system, x0, y0, dt=dt, steps=steps, transient=transient,
    )

    # 6. Homoclinic orbit detection (for saddle points)
    homoclinic = {}
    for i, eq in enumerate(equilibria):
        if eq["classification"] == "saddle":
            homoclinic[i] = detect_homoclinic(
                system, eq["point"], steps=min(steps, 10000), dt=dt,
            )

    return {
        "equilibria": equilibria,
        "trajectory": traj,
        "limit_cycle": is_cycle,
        "lyapunov_exponent": lyap_max,
        "lyapunov_spectrum": lyap_spectrum,
        "lyapunov_convergence": (lyap_times, lyap_estimates),
        "attractor": attractor,
        "homoclinic": homoclinic,
    }
