"""Tests for Chapter 5 dissipative system visualization functions.

All tests use the non-interactive Agg backend so they run headless.
"""

import numpy as np
import pytest
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from nonlinear.dynamical_systems import (
    AutonomousSystem2D,
    DampedPendulum,
    LotkaVolterra,
    simulate,
    flow_lyapunov_spectrum,
)
from nonlinear.visualization.dissipative_plots import (
    plot_dissipative_phase_portrait,
    plot_dissipative_time_series,
    plot_flow_lyapunov_convergence,
    plot_lyapunov_spectrum_bar,
    plot_contraction_field,
    plot_hopf_diagram,
    plot_chapter5_summary,
)


@pytest.fixture
def damped_pendulum():
    return DampedPendulum(b=0.5)


@pytest.fixture
def dp_trajectory(damped_pendulum):
    return simulate(damped_pendulum, 2.0, 0.0, steps=3000, dt=0.01)


# --- Phase portrait ---


class TestPhasePortrait:
    def test_returns_axes(self, damped_pendulum, dp_trajectory):
        ax = plot_dissipative_phase_portrait(
            damped_pendulum, trajectories=[dp_trajectory],
            x_range=(-4, 4), y_range=(-3, 3),
        )
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_with_fixed_points(self, damped_pendulum, dp_trajectory):
        ax = plot_dissipative_phase_portrait(
            damped_pendulum, trajectories=[dp_trajectory],
            x_range=(-4, 4), y_range=(-3, 3),
            fixed_point_guesses=[(0, 0), (3.14, 0)],
        )
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_custom_axes(self, damped_pendulum):
        _, ax_in = plt.subplots()
        ax = plot_dissipative_phase_portrait(
            damped_pendulum, ax=ax_in,
            x_range=(-4, 4), y_range=(-3, 3),
        )
        assert ax is ax_in
        plt.close("all")


# --- Time series ---


class TestTimeSeries:
    def test_returns_axes(self, dp_trajectory):
        ax = plot_dissipative_time_series(dp_trajectory)
        assert isinstance(ax, plt.Axes)
        assert len(ax.get_lines()) == 2
        plt.close("all")

    def test_custom_labels(self, dp_trajectory):
        ax = plot_dissipative_time_series(
            dp_trajectory, labels=("theta", "omega"),
        )
        legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
        assert "theta" in legend_texts
        plt.close("all")


# --- Lyapunov convergence ---


class TestLyapunovConvergence:
    def test_returns_axes(self, damped_pendulum):
        ax = plot_flow_lyapunov_convergence(
            damped_pendulum, 1.0, 0.0,
            steps=1000, transient=100,
        )
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_has_lines(self, damped_pendulum):
        ax = plot_flow_lyapunov_convergence(
            damped_pendulum, 1.0, 0.0,
            steps=1000, transient=100,
        )
        assert len(ax.get_lines()) >= 2  # main + zero line
        plt.close("all")


# --- Lyapunov spectrum bar chart ---


class TestSpectrumBar:
    def test_returns_axes(self):
        spec = np.array([0.1, -0.3])
        ax = plot_lyapunov_spectrum_bar(spec)
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_bar_count(self):
        spec = np.array([0.1, -0.3])
        ax = plot_lyapunov_spectrum_bar(spec)
        assert len(ax.patches) == 2
        plt.close("all")


# --- Contraction field ---


class TestContractionField:
    def test_returns_axes(self, damped_pendulum):
        ax = plot_contraction_field(
            damped_pendulum, n_grid=10,
            x_range=(-2, 2), y_range=(-2, 2),
        )
        assert isinstance(ax, plt.Axes)
        plt.close("all")


# --- Hopf diagram ---


def _make_hopf_system(mu):
    def f(x, y, p):
        return p["mu"] * x - y - x * (x**2 + y**2)

    def g(x, y, p):
        return x + p["mu"] * y - y * (x**2 + y**2)

    return AutonomousSystem2D(f, g, {"mu": mu})


class TestHopfDiagram:
    def test_returns_axes(self):
        params = np.linspace(-0.5, 0.5, 11)
        ax = plot_hopf_diagram(
            _make_hopf_system, params,
            steps=3000, transient=1000,
        )
        assert isinstance(ax, plt.Axes)
        plt.close("all")


# --- Chapter 5 summary ---


class TestChapter5Summary:
    def test_returns_figure_and_axes(self, damped_pendulum):
        fig, axes = plot_chapter5_summary(
            damped_pendulum, 2.0, 0.0,
            dt=0.01, steps=3000, transient=500,
            x_range=(-4, 4), y_range=(-3, 3),
            fixed_point_guesses=[(0, 0)],
        )
        assert isinstance(fig, plt.Figure)
        assert len(axes) == 6
        plt.close("all")

    def test_lotka_volterra_summary(self):
        sys = LotkaVolterra()
        fig, axes = plot_chapter5_summary(
            sys, 1.0, 1.0,
            dt=0.005, steps=5000, transient=500,
            x_range=(0, 4), y_range=(0, 4),
            fixed_point_guesses=[(3, 1.5), (0, 0)],
        )
        assert isinstance(fig, plt.Figure)
        plt.close("all")
