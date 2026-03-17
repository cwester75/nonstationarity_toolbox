"""Hopf bifurcation detection for 2-D autonomous systems.

A Hopf bifurcation occurs when a pair of complex-conjugate eigenvalues
of the Jacobian at a fixed point crosses the imaginary axis as a
parameter varies.  This signals the birth (or death) of a limit cycle.
"""

from __future__ import annotations

import numpy as np

from .fixed_points import jacobian, find_fixed_point


def detect_hopf_bifurcation(
    system_factory,
    parameter_values: np.ndarray,
    fixed_point_guess: tuple[float, float] = (0.0, 0.0),
    tol: float = 1e-6,
) -> list[dict]:
    """Detect Hopf bifurcations across a parameter sweep.

    Parameters
    ----------
    system_factory : callable
        A function ``system_factory(mu) -> AutonomousSystem2D`` that
        creates a system instance for parameter value *mu*.
    parameter_values : array-like
        1-D array of parameter values to sweep.
    fixed_point_guess : tuple of float
        Initial guess for the fixed point (tracked across the sweep).
    tol : float
        Tolerance for zero-crossing of the real part.

    Returns
    -------
    list of dict
        Each dict has keys:
            ``'parameter'`` – parameter value at the bifurcation,
            ``'fixed_point'`` – location of the equilibrium,
            ``'eigenvalues'`` – eigenvalues at the crossing,
            ``'type'`` – ``'supercritical'`` or ``'subcritical'``
                         (based on direction of real-part crossing).
    """
    parameter_values = np.asarray(parameter_values, dtype=float)
    bifurcations: list[dict] = []

    prev_real = None
    prev_imag = False
    guess = np.array(fixed_point_guess, dtype=float)

    for mu in parameter_values:
        system = system_factory(mu)

        fp = find_fixed_point(system, guess)
        if fp is None:
            continue
        guess = fp  # track the equilibrium

        J = jacobian(system, fp[0], fp[1])
        eigs = np.linalg.eigvals(J)

        # Check for complex eigenvalues
        has_imag = any(abs(e.imag) > tol for e in eigs)
        max_real = max(e.real for e in eigs)

        if prev_real is not None and has_imag and prev_imag:
            # Detect sign change in real part
            if prev_real * max_real < 0:
                bif_type = (
                    "supercritical" if prev_real < 0 else "subcritical"
                )
                bifurcations.append({
                    "parameter": float(mu),
                    "fixed_point": fp.copy(),
                    "eigenvalues": eigs.copy(),
                    "type": bif_type,
                })

        prev_real = max_real
        prev_imag = has_imag

    return bifurcations


def hopf_normal_form_coefficient(
    system,
    fixed_point: tuple[float, float],
    eps: float = 1e-4,
) -> float:
    """Estimate the first Lyapunov coefficient of a Hopf bifurcation.

    A negative value indicates a supercritical bifurcation (stable
    limit cycle born), positive indicates subcritical (unstable cycle).

    Uses a finite-difference estimate of the cubic normal-form
    coefficient from the third-order Taylor expansion of the vector
    field at the equilibrium.

    Parameters
    ----------
    system : AutonomousSystem2D
    fixed_point : tuple of float
        The equilibrium point.
    eps : float
        Step size for finite differences.

    Returns
    -------
    float
        Estimated first Lyapunov coefficient.
    """
    x0, y0 = float(fixed_point[0]), float(fixed_point[1])
    f = lambda x, y: system.vector_field(x, y)

    # Second-order partial derivatives via central differences
    f00 = f(x0, y0)
    fxx = (f(x0 + eps, y0) - 2 * f00 + f(x0 - eps, y0)) / eps**2
    fyy = (f(x0, y0 + eps) - 2 * f00 + f(x0, y0 - eps)) / eps**2
    fxy = (
        f(x0 + eps, y0 + eps)
        - f(x0 + eps, y0 - eps)
        - f(x0 - eps, y0 + eps)
        + f(x0 - eps, y0 - eps)
    ) / (4 * eps**2)

    # Third-order partial derivatives
    fxxx = (
        f(x0 + 2 * eps, y0) - 2 * f(x0 + eps, y0)
        + 2 * f(x0 - eps, y0) - f(x0 - 2 * eps, y0)
    ) / (2 * eps**3)
    fyyy = (
        f(x0, y0 + 2 * eps) - 2 * f(x0, y0 + eps)
        + 2 * f(x0, y0 - eps) - f(x0, y0 - 2 * eps)
    ) / (2 * eps**3)
    fxxy = (
        f(x0 + eps, y0 + eps) - 2 * f(x0, y0 + eps) + f(x0 - eps, y0 + eps)
        - f(x0 + eps, y0 - eps) + 2 * f(x0, y0 - eps) - f(x0 - eps, y0 - eps)
    ) / (2 * eps**3)
    fxyy = (
        f(x0 + eps, y0 + eps) - f(x0 + eps, y0 - eps)
        - 2 * f(x0, y0 + eps) + 2 * f(x0, y0 - eps)
        + f(x0 - eps, y0 + eps) - f(x0 - eps, y0 - eps)
    ) / (2 * eps**3)

    J = jacobian(system, x0, y0)
    eigs = np.linalg.eigvals(J)
    omega = abs(eigs[0].imag)

    if omega < 1e-12:
        return 0.0

    # Simplified first Lyapunov coefficient estimate (Kuznetsov formula)
    # l1 = (1/16) * (f_{xxx} + f_{xyy} + g_{xxy} + g_{yyy})
    #    + (1/(16*omega)) * (f_{xy}*(f_{xx}+f_{yy}) - g_{xy}*(g_{xx}+g_{yy})
    #                        - f_{xx}*g_{xx} + f_{yy}*g_{yy})
    l1 = (1.0 / 16.0) * (
        fxxx[0] + fxyy[0] + fxxy[1] + fyyy[1]
    ) + (1.0 / (16.0 * omega)) * (
        fxy[0] * (fxx[0] + fyy[0])
        - fxy[1] * (fxx[1] + fyy[1])
        - fxx[0] * fxx[1]
        + fyy[0] * fyy[1]
    )

    return float(l1)
