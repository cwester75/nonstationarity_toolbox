"""Example Hamiltonian systems.

Provides factory functions that return HamiltonianSystem instances
for commonly studied models:
    - Harmonic oscillator
    - Simple pendulum
    - Hénon-Heiles system (2 DOF, classic chaos model)
    - Double-well potential
"""

from __future__ import annotations

import numpy as np

from .system import HamiltonianSystem


def harmonic_oscillator(m: float = 1.0, k: float = 1.0) -> HamiltonianSystem:
    """Create a 1-DOF harmonic oscillator.

    H = p²/(2m) + kq²/2

    Parameters
    ----------
    m : float
        Mass.
    k : float
        Spring constant.

    Returns
    -------
    HamiltonianSystem
    """
    def H(q, p):
        return p[0] ** 2 / (2 * m) + 0.5 * k * q[0] ** 2

    return HamiltonianSystem(H, ndof=1)


def pendulum(m: float = 1.0, g: float = 9.81, length: float = 1.0) -> HamiltonianSystem:
    """Create a 1-DOF simple pendulum.

    H = p²/(2ml²) + mgl(1 - cos q)

    Parameters
    ----------
    m : float
        Mass.
    g : float
        Gravitational acceleration.
    length : float
        Pendulum length.

    Returns
    -------
    HamiltonianSystem
    """
    ml2 = m * length ** 2

    def H(q, p):
        return p[0] ** 2 / (2 * ml2) + m * g * length * (1 - np.cos(q[0]))

    return HamiltonianSystem(H, ndof=1)


def henon_heiles() -> HamiltonianSystem:
    """Create a 2-DOF Hénon-Heiles system.

    H = (px² + py²)/2 + (x² + y²)/2 + x²y - y³/3

    This is a classic model exhibiting the transition from integrable
    to chaotic motion as energy increases (chaos onset near E ≈ 1/6).

    Returns
    -------
    HamiltonianSystem
    """
    def H(q, p):
        x, y = q[0], q[1]
        px, py = p[0], p[1]
        kinetic = 0.5 * (px ** 2 + py ** 2)
        potential = 0.5 * (x ** 2 + y ** 2) + x ** 2 * y - y ** 3 / 3
        return kinetic + potential

    return HamiltonianSystem(H, ndof=2)


def double_well(a: float = 1.0, b: float = 1.0) -> HamiltonianSystem:
    """Create a 1-DOF double-well potential system.

    H = p²/2 + a*q⁴/4 - b*q²/2

    Parameters
    ----------
    a : float
        Quartic coefficient.
    b : float
        Quadratic coefficient.

    Returns
    -------
    HamiltonianSystem
    """
    def H(q, p):
        return p[0] ** 2 / 2 + a * q[0] ** 4 / 4 - b * q[0] ** 2 / 2

    return HamiltonianSystem(H, ndof=1)
