"""Tests for Chapter 4 Hamiltonian visualization functions.

All tests use the non-interactive Agg backend so they run headless.
They verify that plotting functions return the expected objects and
produce graphical elements without raising exceptions.
"""

import numpy as np
import pytest
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from nonlinear.hamiltonian import (
    HamiltonianSystem,
    harmonic_oscillator,
    pendulum,
    henon_heiles,
    integrate_hamiltonian,
    compute_poincare_section,
)
from nonlinear.visualization.hamiltonian_plots import (
    plot_phase_space,
    plot_poincare_section as plot_poincare,
    plot_energy_drift,
    plot_multi_trajectory_phase_space,
    plot_energy_contours,
    plot_time_series,
    plot_lyapunov_convergence,
    plot_phase_density,
    plot_chapter4_summary,
)


@pytest.fixture
def ho_trajectory():
    """Short harmonic oscillator trajectory for testing."""
    sys = harmonic_oscillator()
    q0 = np.array([1.0])
    p0 = np.array([0.0])
    q, p = integrate_hamiltonian(sys, q0, p0, dt=0.01, steps=2000)
    return sys, q, p


@pytest.fixture
def hh_trajectory():
    """Short Hénon-Heiles trajectory for testing."""
    sys = henon_heiles()
    q0 = np.array([0.0, 0.1])
    p0 = np.array([0.3, 0.0])
    q, p = integrate_hamiltonian(sys, q0, p0, dt=0.01, steps=5000)
    return sys, q, p


# --- Energy contours ---


class TestEnergyContours:
    def test_returns_axes(self, ho_trajectory):
        sys, _, _ = ho_trajectory
        ax = plot_energy_contours(sys, n_grid=30, n_levels=10)
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_custom_axes(self, ho_trajectory):
        sys, _, _ = ho_trajectory
        _, ax_in = plt.subplots()
        ax = plot_energy_contours(sys, ax=ax_in, n_grid=30)
        assert ax is ax_in
        plt.close("all")

    def test_multidof_system(self, hh_trajectory):
        sys, _, _ = hh_trajectory
        ax = plot_energy_contours(sys, n_grid=30, dof_index=0)
        assert isinstance(ax, plt.Axes)
        plt.close("all")


# --- Time series ---


class TestTimeSeries:
    def test_returns_axes(self, ho_trajectory):
        _, q, p = ho_trajectory
        ax = plot_time_series(q, p)
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_with_momentum(self, ho_trajectory):
        _, q, p = ho_trajectory
        ax = plot_time_series(q, p, p_indices=[0])
        assert isinstance(ax, plt.Axes)
        # Should have 2 lines (q0 + p0)
        assert len(ax.get_lines()) == 2
        plt.close("all")

    def test_multidof(self, hh_trajectory):
        _, q, p = hh_trajectory
        ax = plot_time_series(q, p, q_indices=[0, 1])
        assert isinstance(ax, plt.Axes)
        assert len(ax.get_lines()) == 2
        plt.close("all")


# --- Lyapunov convergence ---


class TestLyapunovConvergence:
    def test_returns_axes(self):
        sys = harmonic_oscillator()
        ax = plot_lyapunov_convergence(
            sys, np.array([1.0]), np.array([0.0]),
            steps=1000, transient=100, renorm_interval=10,
        )
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_custom_axes(self):
        sys = harmonic_oscillator()
        _, ax_in = plt.subplots()
        ax = plot_lyapunov_convergence(
            sys, np.array([1.0]), np.array([0.0]),
            steps=500, transient=50, ax=ax_in,
        )
        assert ax is ax_in
        plt.close("all")

    def test_has_convergence_line(self):
        sys = harmonic_oscillator()
        ax = plot_lyapunov_convergence(
            sys, np.array([1.0]), np.array([0.0]),
            steps=1000, transient=100,
        )
        # Should have main curve + zero line + final value line
        assert len(ax.get_lines()) >= 2
        plt.close("all")


# --- Phase density (Liouville) ---


class TestPhaseDensity:
    def test_returns_figure_and_axes(self):
        sys = harmonic_oscillator()
        fig, axes = plot_phase_density(
            sys, np.array([1.0]), np.array([0.0]),
            n_particles=20, n_snapshots=3, steps_per_snapshot=50,
        )
        assert isinstance(fig, plt.Figure)
        assert len(axes) == 3
        plt.close("all")

    def test_four_snapshots(self):
        sys = harmonic_oscillator()
        fig, axes = plot_phase_density(
            sys, np.array([1.0]), np.array([0.0]),
            n_particles=10, n_snapshots=4, steps_per_snapshot=30,
        )
        assert len(axes) == 4
        plt.close("all")

    def test_multidof(self):
        sys = henon_heiles()
        fig, axes = plot_phase_density(
            sys, np.array([0.0, 0.1]), np.array([0.2, 0.0]),
            n_particles=10, n_snapshots=2, steps_per_snapshot=30,
        )
        assert len(axes) == 2
        plt.close("all")


# --- Chapter 4 summary ---


class TestChapter4Summary:
    def test_1dof_summary(self):
        sys = harmonic_oscillator()
        fig, axes = plot_chapter4_summary(
            sys, np.array([1.0]), np.array([0.0]),
            dt=0.01, steps=2000,
        )
        assert isinstance(fig, plt.Figure)
        assert len(axes) == 6
        plt.close("all")

    def test_2dof_summary(self):
        sys = henon_heiles()
        fig, axes = plot_chapter4_summary(
            sys, np.array([0.0, 0.1]), np.array([0.3, 0.0]),
            dt=0.01, steps=5000,
        )
        assert isinstance(fig, plt.Figure)
        assert len(axes) == 6
        plt.close("all")


# --- Existing plots still work ---


class TestExistingPlots:
    def test_phase_space(self, ho_trajectory):
        _, q, p = ho_trajectory
        ax = plot_phase_space(q, p)
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_energy_drift(self, ho_trajectory):
        sys, q, p = ho_trajectory
        ax = plot_energy_drift(sys, q, p)
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_multi_trajectory(self, ho_trajectory):
        sys, q, p = ho_trajectory
        ax = plot_multi_trajectory_phase_space([(q, p), (q, p)])
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_poincare_section(self, hh_trajectory):
        _, q, p = hh_trajectory
        section = compute_poincare_section(q, p, section_index=1)
        if len(section) > 0:
            ax = plot_poincare(section)
            assert isinstance(ax, plt.Axes)
        plt.close("all")
