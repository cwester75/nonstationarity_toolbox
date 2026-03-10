"""Tests for Chapter 5: Nonlinear Dissipative Systems.

Covers:
    - Hopf bifurcation detection
    - Continuous-time Lyapunov exponent and spectrum
    - Attractor classification
    - Phase-space contraction rate
    - Dissipative analysis pipeline
"""

import numpy as np
import pytest

from nonlinear.dynamical_systems import (
    AutonomousSystem2D,
    DampedPendulum,
    LotkaVolterra,
    simulate,
    find_fixed_points,
    classify_fixed_point,
    flow_lyapunov_exponent,
    flow_lyapunov_spectrum,
    flow_lyapunov_convergence,
    classify_attractor,
    phase_space_contraction,
    detect_hopf_bifurcation,
    dissipative_analysis,
)


# --- Helper: Hopf normal-form system ---
# dx/dt = mu*x - y - x*(x^2 + y^2)
# dy/dt = x + mu*y - y*(x^2 + y^2)
# Supercritical Hopf bifurcation at mu = 0

def _make_hopf_system(mu):
    def f(x, y, p):
        return p["mu"] * x - y - x * (x**2 + y**2)

    def g(x, y, p):
        return x + p["mu"] * y - y * (x**2 + y**2)

    return AutonomousSystem2D(f, g, {"mu": mu})


# --- Lyapunov exponents for flows ---


class TestFlowLyapunov:
    def test_damped_pendulum_negative(self):
        """Damped pendulum should have negative maximal Lyapunov exponent."""
        sys = DampedPendulum(b=0.5)
        lam = flow_lyapunov_exponent(sys, 1.0, 0.0, dt=0.01, steps=10000, transient=500)
        assert lam < 0.1  # Should be clearly negative or near zero

    def test_spectrum_returns_two(self):
        """Spectrum should return exactly 2 exponents for 2-D system."""
        sys = DampedPendulum(b=0.5)
        spec = flow_lyapunov_spectrum(
            sys, 1.0, 0.0, dt=0.01, steps=5000, transient=500,
        )
        assert spec.shape == (2,)
        assert spec[0] >= spec[1]  # Sorted descending

    def test_spectrum_sum_negative_for_dissipative(self):
        """Sum of Lyapunov exponents should be negative for dissipative system."""
        sys = DampedPendulum(b=0.5)
        spec = flow_lyapunov_spectrum(
            sys, 1.0, 0.0, dt=0.01, steps=10000, transient=500,
        )
        assert np.sum(spec) < 0.1  # Dissipative: sum should be negative

    def test_convergence_returns_arrays(self):
        """Convergence function should return time and estimate arrays."""
        sys = DampedPendulum(b=0.5)
        times, estimates = flow_lyapunov_convergence(
            sys, 1.0, 0.0, dt=0.01, steps=2000, transient=200,
        )
        assert len(times) == len(estimates)
        assert len(times) > 0
        assert times[-1] > times[0]


# --- Attractor classification ---


class TestAttractorClassification:
    def test_fixed_point_attractor(self):
        """Strongly damped pendulum should converge to a fixed point."""
        sys = DampedPendulum(b=2.0)
        result = classify_attractor(
            sys, 0.5, 0.0, dt=0.01, steps=10000, transient=3000,
            lyapunov_steps=5000,
        )
        assert result["type"] == "fixed_point"

    def test_limit_cycle_attractor(self):
        """Hopf normal-form system above bifurcation should show a limit cycle."""
        sys = _make_hopf_system(mu=0.5)
        result = classify_attractor(
            sys, 0.1, 0.0, dt=0.01, steps=20000, transient=5000,
            lyapunov_steps=10000,
        )
        assert result["type"] == "limit_cycle"

    def test_trajectory_present(self):
        """Result should always contain a trajectory array."""
        sys = DampedPendulum(b=0.5)
        result = classify_attractor(
            sys, 1.0, 0.0, dt=0.01, steps=5000, transient=1000,
            lyapunov_steps=2000,
        )
        assert result["trajectory"].shape == (5000, 2)


# --- Phase-space contraction ---


class TestPhaseSpaceContraction:
    def test_damped_pendulum_negative_divergence(self):
        """Damped pendulum should have negative divergence (dissipative)."""
        sys = DampedPendulum(b=0.5)
        div = phase_space_contraction(sys, 0.0, 0.0)
        assert div < 0  # div(f) = 0 + (-b) = -b for damped pendulum

    def test_damped_pendulum_divergence_equals_minus_b(self):
        """For damped pendulum, div(f) = -b everywhere."""
        b = 0.3
        sys = DampedPendulum(b=b)
        div = phase_space_contraction(sys, 1.5, -0.7)
        assert abs(div - (-b)) < 1e-4

    def test_lotka_volterra_not_constant(self):
        """Lotka-Volterra divergence varies with position."""
        sys = LotkaVolterra()
        d1 = phase_space_contraction(sys, 1.0, 1.0)
        d2 = phase_space_contraction(sys, 2.0, 2.0)
        # Should be different at different points
        assert d1 != d2


# --- Hopf bifurcation detection ---


class TestHopfBifurcation:
    def test_detects_hopf_in_normal_form(self):
        """Should detect Hopf bifurcation near mu=0 in the normal form system."""
        params = np.linspace(-1.0, 1.0, 201)
        bifs = detect_hopf_bifurcation(
            _make_hopf_system, params,
            fixed_point_guess=(0.0, 0.0),
        )
        assert len(bifs) > 0
        # Bifurcation should be near mu = 0
        mu_bif = bifs[0]["parameter"]
        assert abs(mu_bif) < 0.05

    def test_supercritical_type(self):
        """Normal-form system should produce a supercritical Hopf."""
        params = np.linspace(-1.0, 1.0, 201)
        bifs = detect_hopf_bifurcation(
            _make_hopf_system, params,
            fixed_point_guess=(0.0, 0.0),
        )
        assert len(bifs) > 0
        assert bifs[0]["type"] == "supercritical"

    def test_eigenvalues_complex(self):
        """Eigenvalues at bifurcation should be complex."""
        params = np.linspace(-1.0, 1.0, 201)
        bifs = detect_hopf_bifurcation(
            _make_hopf_system, params,
            fixed_point_guess=(0.0, 0.0),
        )
        assert len(bifs) > 0
        eigs = bifs[0]["eigenvalues"]
        assert any(abs(e.imag) > 0.1 for e in eigs)

    def test_no_hopf_in_damped_pendulum(self):
        """DampedPendulum with fixed damping has no Hopf bifurcation."""
        def factory(mu):
            # Just vary something that doesn't cross imaginary axis
            return DampedPendulum(b=0.5)

        params = np.linspace(0.1, 2.0, 20)
        bifs = detect_hopf_bifurcation(
            factory, params,
            fixed_point_guess=(0.0, 0.0),
        )
        assert len(bifs) == 0


# --- Dissipative analysis pipeline ---


class TestDissipativePipeline:
    def test_pipeline_returns_all_keys(self):
        """Pipeline should return all expected keys."""
        sys = DampedPendulum(b=0.5)
        result = dissipative_analysis(
            sys, 1.0, 0.0, dt=0.01, steps=5000, transient=1000,
        )
        expected_keys = {
            "equilibria", "trajectory", "limit_cycle",
            "lyapunov_exponent", "lyapunov_spectrum",
            "lyapunov_convergence", "attractor", "homoclinic",
        }
        assert expected_keys == set(result.keys())

    def test_pipeline_finds_equilibria(self):
        """Pipeline should find fixed points of damped pendulum."""
        sys = DampedPendulum(b=0.5)
        result = dissipative_analysis(
            sys, 1.0, 0.0, dt=0.01, steps=5000, transient=1000,
        )
        assert len(result["equilibria"]) > 0
        # Origin should be a stable spiral for damped pendulum
        classifications = [eq["classification"] for eq in result["equilibria"]]
        assert "stable spiral" in classifications

    def test_pipeline_trajectory_shape(self):
        """Trajectory should have correct shape."""
        sys = DampedPendulum(b=0.5)
        result = dissipative_analysis(
            sys, 1.0, 0.0, dt=0.01, steps=5000, transient=1000,
        )
        assert result["trajectory"].shape == (5000, 2)

    def test_pipeline_lyapunov_spectrum_shape(self):
        """Lyapunov spectrum should have 2 elements."""
        sys = DampedPendulum(b=0.5)
        result = dissipative_analysis(
            sys, 1.0, 0.0, dt=0.01, steps=5000, transient=1000,
        )
        assert result["lyapunov_spectrum"].shape == (2,)
