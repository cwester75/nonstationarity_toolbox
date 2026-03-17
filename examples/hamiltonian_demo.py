"""Chapter 4 — Nonlinear Hamiltonian Systems: Demonstration Script.

This script demonstrates the key components of the Hamiltonian systems
module:
    1. Harmonic oscillator phase portrait
    2. Hénon-Heiles trajectory and Poincaré section
    3. Energy conservation verification
    4. Chaos detection via Lyapunov exponent
"""

import numpy as np
from nonlinear.hamiltonian import (
    harmonic_oscillator,
    henon_heiles,
    integrate_hamiltonian,
    compute_poincare_section,
    check_energy_conservation,
    hamiltonian_lyapunov,
    classify_poincare_dynamics,
    compute_action_variable,
)

# --- 1. Harmonic oscillator ---
print("=" * 60)
print("1. Harmonic Oscillator")
print("=" * 60)

ho = harmonic_oscillator(m=1.0, k=1.0)
q0 = np.array([1.0])
p0 = np.array([0.0])
q_traj, p_traj = integrate_hamiltonian(ho, q0, p0, dt=0.01, steps=10000)

conserved, variation = check_energy_conservation(ho, q_traj, p_traj)
print(f"  Initial energy: {ho.energy(q0, p0):.6f}")
print(f"  Energy conserved: {conserved} (variation: {variation:.2e})")

# Action variable from one orbit (first period, T=2π for ω=1)
period_steps = int(2 * np.pi / 0.01)
I = compute_action_variable(q_traj[:period_steps, 0], p_traj[:period_steps, 0])
print(f"  Action variable I = {I:.4f} (expected 0.5)")

# Lyapunov exponent
lam = hamiltonian_lyapunov(ho, q0, p0, dt=0.01, steps=10000, transient=500)
print(f"  Lyapunov exponent: {lam:.4f} (expected ~0)")

# --- 2. Hénon-Heiles: regular regime ---
print()
print("=" * 60)
print("2. Hénon-Heiles — Low Energy (Regular)")
print("=" * 60)

hh = henon_heiles()
q0_reg = np.array([0.0, 0.0])
p0_reg = np.array([0.2, 0.2])
E_reg = hh.energy(q0_reg, p0_reg)
print(f"  Energy: {E_reg:.4f} (chaos onset ~0.1667)")

q_traj, p_traj = integrate_hamiltonian(hh, q0_reg, p0_reg, dt=0.005, steps=100000)
conserved, variation = check_energy_conservation(hh, q_traj, p_traj, tolerance=1e-4)
print(f"  Energy conserved: {conserved} (variation: {variation:.2e})")

section = compute_poincare_section(q_traj, p_traj, section_index=1)
dynamics = classify_poincare_dynamics(section)
print(f"  Poincaré section crossings: {len(section)}")
print(f"  Dynamics classification: {dynamics}")

# --- 3. Hénon-Heiles: chaotic regime ---
print()
print("=" * 60)
print("3. Hénon-Heiles — High Energy (Chaotic)")
print("=" * 60)

p0_chaos = np.array([0.45, 0.15])
E_chaos = hh.energy(q0_reg, p0_chaos)
print(f"  Energy: {E_chaos:.4f}")

q_traj_c, p_traj_c = integrate_hamiltonian(
    hh, q0_reg, p0_chaos, dt=0.005, steps=100000
)
conserved_c, var_c = check_energy_conservation(
    hh, q_traj_c, p_traj_c, tolerance=1e-3
)
print(f"  Energy conserved: {conserved_c} (variation: {var_c:.2e})")

section_c = compute_poincare_section(q_traj_c, p_traj_c, section_index=1)
dynamics_c = classify_poincare_dynamics(section_c)
print(f"  Poincaré section crossings: {len(section_c)}")
print(f"  Dynamics classification: {dynamics_c}")

print()
print("=" * 60)
print("Chapter 4 demonstration complete.")
print("=" * 60)
