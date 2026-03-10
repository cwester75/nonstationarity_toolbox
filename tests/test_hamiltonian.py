"""Tests for the Hamiltonian systems module (Chapter 4)."""

import numpy as np
import pytest

from nonlinear.hamiltonian import (
    HamiltonianSystem,
    integrate_hamiltonian,
    leapfrog_step,
    stormer_verlet_step,
    symplectic_euler_step,
    compute_poincare_section,
    interpolate_crossing,
    compute_action_variable,
    compute_angle_variable,
    check_energy_conservation,
    check_integrability,
    hamiltonian_lyapunov,
    classify_poincare_dynamics,
    harmonic_oscillator,
    pendulum,
    henon_heiles,
    double_well,
)


# --- HamiltonianSystem ---


def test_harmonic_oscillator_energy():
    """Harmonic oscillator energy should be p²/2 + q²/2 for unit mass/spring."""
    sys = harmonic_oscillator(m=1.0, k=1.0)
    q = np.array([1.0])
    p = np.array([0.0])
    assert abs(sys.energy(q, p) - 0.5) < 1e-10


def test_harmonic_oscillator_equations():
    """dq/dt = p/m and dp/dt = -kq for harmonic oscillator."""
    sys = harmonic_oscillator(m=1.0, k=1.0)
    q = np.array([2.0])
    p = np.array([3.0])
    dq = sys.dq_dt(q, p)
    dp = sys.dp_dt(q, p)
    assert abs(dq[0] - 3.0) < 1e-5
    assert abs(dp[0] - (-2.0)) < 1e-5


def test_pendulum_energy_at_rest():
    """Pendulum at bottom has zero energy."""
    sys = pendulum(m=1.0, g=9.81, length=1.0)
    q = np.array([0.0])
    p = np.array([0.0])
    assert abs(sys.energy(q, p)) < 1e-10


def test_henon_heiles_energy_at_origin():
    """Hénon-Heiles energy at origin with zero momenta is zero."""
    sys = henon_heiles()
    q = np.array([0.0, 0.0])
    p = np.array([0.0, 0.0])
    assert abs(sys.energy(q, p)) < 1e-10


def test_double_well_two_minima():
    """Double-well potential has minima at ±√(b/a)."""
    sys = double_well(a=1.0, b=1.0)
    # Minimum at q = 1.0
    e_min = sys.energy(np.array([1.0]), np.array([0.0]))
    e_origin = sys.energy(np.array([0.0]), np.array([0.0]))
    assert e_min < e_origin


# --- Symplectic integrators ---


def test_verlet_energy_conservation_harmonic():
    """Verlet integrator should conserve energy for harmonic oscillator."""
    sys = harmonic_oscillator()
    q0 = np.array([1.0])
    p0 = np.array([0.0])
    q_traj, p_traj = integrate_hamiltonian(sys, q0, p0, dt=0.01, steps=10000)
    conserved, variation = check_energy_conservation(sys, q_traj, p_traj, tolerance=1e-4)
    assert conserved
    assert variation < 1e-4


def test_euler_less_accurate_than_verlet():
    """Symplectic Euler should drift more than Verlet."""
    sys = harmonic_oscillator()
    q0 = np.array([1.0])
    p0 = np.array([0.0])
    _, p_v = integrate_hamiltonian(sys, q0, p0, dt=0.05, steps=2000, method="verlet")
    q_v, _ = integrate_hamiltonian(sys, q0, p0, dt=0.05, steps=2000, method="verlet")
    _, var_v = check_energy_conservation(sys, q_v, p_v)

    q_e, p_e = integrate_hamiltonian(sys, q0, p0, dt=0.05, steps=2000, method="euler")
    _, var_e = check_energy_conservation(sys, q_e, p_e)
    # Verlet should be more accurate (or at least not worse)
    assert var_v <= var_e + 1e-12


def test_integrate_unknown_method_raises():
    """Unknown method should raise ValueError."""
    sys = harmonic_oscillator()
    with pytest.raises(ValueError, match="Unknown method"):
        integrate_hamiltonian(sys, np.array([1.0]), np.array([0.0]), method="rk4")


def test_leapfrog_is_verlet():
    """leapfrog_step and stormer_verlet_step should be identical."""
    assert leapfrog_step is stormer_verlet_step


# --- Poincaré section ---


def test_poincare_section_henon_heiles():
    """Poincaré section of Hénon-Heiles should produce crossings."""
    sys = henon_heiles()
    q0 = np.array([0.0, 0.0])
    p0 = np.array([0.3, 0.1])
    q_traj, p_traj = integrate_hamiltonian(sys, q0, p0, dt=0.01, steps=50000)
    section = compute_poincare_section(q_traj, p_traj, section_index=1)
    assert len(section) > 0
    # Section should have 3 columns (q0, p0, p1) for 2-DOF
    assert section.shape[1] == 3


def test_poincare_section_no_crossings():
    """No crossings should return empty array."""
    # Harmonic oscillator with section at q=10 (never reached)
    sys = harmonic_oscillator()
    q0 = np.array([1.0])
    p0 = np.array([0.0])
    q_traj, p_traj = integrate_hamiltonian(sys, q0, p0, dt=0.01, steps=1000)
    section = compute_poincare_section(
        q_traj, p_traj, section_index=0, section_value=10.0
    )
    assert len(section) == 0


def test_interpolate_crossing():
    """Interpolation should find the zero crossing."""
    s1 = np.array([0.0, -0.5, 1.0, 2.0])
    s2 = np.array([0.1, 0.5, 1.1, 2.1])
    result = interpolate_crossing(s1, s2, -0.5, 0.5)
    assert abs(result[1]) < 1e-10  # The crossing coordinate should be near 0


# --- Action-angle variables ---


def test_action_variable_harmonic():
    """Action variable of harmonic oscillator: I = E/ω = A²/2 for ω=1."""
    sys = harmonic_oscillator(m=1.0, k=1.0)
    A = 1.0
    # Generate one full orbit
    n = 10000
    theta = np.linspace(0, 2 * np.pi, n, endpoint=True)
    q = A * np.cos(theta)
    p = -A * np.sin(theta)
    I = compute_action_variable(q, p)
    # For ω=1: I = E/ω = (A²/2)/1 = 0.5
    assert abs(I - 0.5) < 0.01


def test_angle_variable_mod():
    """Angle variable should wrap modulo 2π."""
    theta = compute_angle_variable(omega=1.0, t=7 * np.pi)
    assert 0 <= theta < 2 * np.pi


# --- Integrability ---


def test_energy_conservation_check():
    """Energy conservation check should pass for well-integrated system."""
    sys = harmonic_oscillator()
    q0 = np.array([1.0])
    p0 = np.array([0.0])
    q_traj, p_traj = integrate_hamiltonian(sys, q0, p0, dt=0.01, steps=5000)
    conserved, _ = check_energy_conservation(sys, q_traj, p_traj, tolerance=1e-4)
    assert conserved


def test_check_integrability_with_energy():
    """The Hamiltonian itself should be detected as a conserved quantity."""
    sys = harmonic_oscillator()
    q0 = np.array([1.0])
    p0 = np.array([0.0])
    q_traj, p_traj = integrate_hamiltonian(sys, q0, p0, dt=0.01, steps=5000)
    result = check_integrability([sys.H], q_traj, p_traj)
    assert len(result) == 1


# --- Chaos detection ---


def test_lyapunov_harmonic_near_zero():
    """Harmonic oscillator should have near-zero Lyapunov exponent."""
    sys = harmonic_oscillator()
    lam = hamiltonian_lyapunov(
        sys, np.array([1.0]), np.array([0.0]),
        dt=0.01, steps=10000, transient=500
    )
    assert abs(lam) < 0.05


def test_classify_regular():
    """Points forming a line should be classified as regular."""
    t = np.linspace(0, 2 * np.pi, 200)
    points = np.column_stack([np.cos(t), np.sin(t) * 0.001])
    result = classify_poincare_dynamics(points)
    assert result == "regular"


def test_classify_chaotic():
    """Randomly scattered points should be classified as chaotic."""
    rng = np.random.default_rng(42)
    points = rng.standard_normal((200, 2))
    result = classify_poincare_dynamics(points)
    assert result == "chaotic"


def test_classify_insufficient():
    """Too few points should return insufficient_data."""
    points = np.array([[0.0, 0.0], [1.0, 1.0]])
    result = classify_poincare_dynamics(points)
    assert result == "insufficient_data"


# --- Integration: full pipeline ---


def test_henon_heiles_energy_bounded():
    """Hénon-Heiles energy should stay bounded for moderate energy."""
    sys = henon_heiles()
    q0 = np.array([0.0, 0.0])
    p0 = np.array([0.2, 0.2])
    q_traj, p_traj = integrate_hamiltonian(sys, q0, p0, dt=0.005, steps=20000)
    conserved, variation = check_energy_conservation(
        sys, q_traj, p_traj, tolerance=1e-4
    )
    assert conserved
