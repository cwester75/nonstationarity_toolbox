"""Tests for Chapter 2 & 3 visualization functions.

All tests use the non-interactive Agg backend so they run headless.
They verify that plotting functions return Axes objects and produce
the expected graphical elements without raising exceptions.
"""

import numpy as np
import pytest
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from nonlinear.dynamical_systems import (
    AutonomousSystem2D,
    Pendulum,
    LotkaVolterra,
    simulate,
    find_fixed_points,
    classify_fixed_point,
)
from nonlinear.visualization.autonomous_phase import (
    plot_vector_field,
    plot_streamlines,
    plot_trajectory,
    plot_trajectories,
    plot_fixed_points,
    plot_poincare_section,
    plot_phase_portrait_composite,
)
from nonlinear.visualization.timeseries_viz import (
    plot_recurrence_matrix,
    plot_recurrence_rate_vs_epsilon,
    plot_determinism_vs_epsilon,
    plot_embedding_2d,
    plot_embedding_3d,
    plot_autocorrelation,
    plot_diagnostics_summary,
)
from nonlinear.attractor.recurrence import RecurrenceAnalysis
from nonlinear.core.embedding import PhaseSpace


@pytest.fixture(autouse=True)
def close_figs():
    """Close all figures after every test."""
    yield
    plt.close("all")


# ── Chapter 3: Autonomous systems visualisation ────────────────────


class TestVectorField:
    def test_returns_axes(self):
        pend = Pendulum()
        ax = plot_vector_field(pend, (-3, 3), (-2, 2))
        assert isinstance(ax, plt.Axes)

    def test_custom_axes(self):
        pend = Pendulum()
        fig, ax = plt.subplots()
        out = plot_vector_field(pend, (-3, 3), (-2, 2), ax=ax)
        assert out is ax

    def test_unnormalized(self):
        pend = Pendulum()
        ax = plot_vector_field(pend, (-3, 3), (-2, 2), normalize=False)
        assert isinstance(ax, plt.Axes)


class TestStreamlines:
    def test_returns_axes(self):
        pend = Pendulum()
        ax = plot_streamlines(pend, (-3, 3), (-2, 2))
        assert isinstance(ax, plt.Axes)


class TestTrajectory:
    def test_single_trajectory(self):
        traj = simulate(Pendulum(), 0.5, 0.0, steps=500, dt=0.01)
        ax = plot_trajectory(traj)
        assert isinstance(ax, plt.Axes)
        assert len(ax.lines) >= 1

    def test_multiple_trajectories(self):
        pend = Pendulum()
        t1 = simulate(pend, 0.3, 0.0, steps=500, dt=0.01)
        t2 = simulate(pend, 1.0, 0.0, steps=500, dt=0.01)
        ax = plot_trajectories([t1, t2])
        assert isinstance(ax, plt.Axes)
        assert len(ax.lines) >= 2


class TestFixedPointsPlot:
    def test_with_classifications(self):
        fps = [np.array([0, 0]), np.array([3, 1.5])]
        labels = ["saddle", "center"]
        ax = plot_fixed_points(fps, labels)
        assert isinstance(ax, plt.Axes)

    def test_without_classifications(self):
        fps = [np.array([1, 2])]
        ax = plot_fixed_points(fps)
        assert isinstance(ax, plt.Axes)


class TestPoincareSection:
    def test_returns_axes(self):
        traj = simulate(Pendulum(), 0.5, 0.0, steps=10000, dt=0.01)
        ax = plot_poincare_section(traj)
        assert isinstance(ax, plt.Axes)

    def test_y_section(self):
        traj = simulate(Pendulum(), 0.5, 0.0, steps=10000, dt=0.01)
        ax = plot_poincare_section(traj, section_coord=1, section_value=0.0)
        assert isinstance(ax, plt.Axes)


class TestCompositePortrait:
    def test_full_composite(self):
        lv = LotkaVolterra()
        traj = simulate(lv, 2, 1, steps=5000, dt=0.01)
        fps = find_fixed_points(lv, [(0.1, 0.1), (2.5, 1.0)])
        cls = [classify_fixed_point(lv, fp) for fp in fps]
        ax = plot_phase_portrait_composite(lv, trajectory=traj,
                                           fixed_points=fps,
                                           classifications=cls)
        assert isinstance(ax, plt.Axes)

    def test_field_only(self):
        pend = Pendulum()
        ax = plot_phase_portrait_composite(pend, x_range=(-4, 4),
                                           y_range=(-3, 3))
        assert isinstance(ax, plt.Axes)


# ── Chapter 2: Time-series visualisation ───────────────────────────


def _lorenz_x(n=2000, dt=0.01):
    """Generate x-component of Lorenz attractor for testing."""
    x, y, z = 1.0, 1.0, 1.0
    sigma, rho, beta = 10.0, 28.0, 8 / 3
    xs = np.empty(n)
    for i in range(n):
        xs[i] = x
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        x += dx * dt
        y += dy * dt
        z += dz * dt
    return xs


class TestRecurrencePlot:
    def test_returns_axes(self):
        data = np.sin(np.linspace(0, 4 * np.pi, 200))
        ps = PhaseSpace(data)
        embedded = ps.embed(dimension=2, delay=5)
        ra = RecurrenceAnalysis(embedded)
        R = ra.recurrence_matrix(epsilon=0.5)
        ax = plot_recurrence_matrix(R)
        assert isinstance(ax, plt.Axes)

    def test_custom_axes(self):
        R = np.eye(10, dtype=bool)
        fig, ax = plt.subplots()
        out = plot_recurrence_matrix(R, ax=ax)
        assert out is ax


class TestRecurrenceMetrics:
    def test_rr_vs_epsilon(self):
        data = np.sin(np.linspace(0, 4 * np.pi, 200))
        ps = PhaseSpace(data)
        embedded = ps.embed(dimension=2, delay=5)
        ra = RecurrenceAnalysis(embedded)
        ax = plot_recurrence_rate_vs_epsilon(ra, np.linspace(0.1, 2.0, 10))
        assert isinstance(ax, plt.Axes)

    def test_det_vs_epsilon(self):
        data = np.sin(np.linspace(0, 4 * np.pi, 200))
        ps = PhaseSpace(data)
        embedded = ps.embed(dimension=2, delay=5)
        ra = RecurrenceAnalysis(embedded)
        ax = plot_determinism_vs_epsilon(ra, np.linspace(0.1, 2.0, 10))
        assert isinstance(ax, plt.Axes)


class TestEmbeddingPlots:
    def test_2d(self):
        data = np.sin(np.linspace(0, 10 * np.pi, 500))
        ps = PhaseSpace(data)
        embedded = ps.embed(dimension=2, delay=10)
        ax = plot_embedding_2d(embedded)
        assert isinstance(ax, plt.Axes)

    def test_3d(self):
        data = _lorenz_x(1500)
        ps = PhaseSpace(data)
        embedded = ps.embed(dimension=3, delay=5)
        ax = plot_embedding_3d(embedded)
        # Axes3D is a subclass, just check it's not None
        assert ax is not None


class TestAutocorrelation:
    def test_returns_axes(self):
        data = np.sin(np.linspace(0, 10 * np.pi, 500))
        ax = plot_autocorrelation(data, max_lag=40)
        assert isinstance(ax, plt.Axes)

    def test_with_delay_marker(self):
        data = np.sin(np.linspace(0, 10 * np.pi, 500))
        ax = plot_autocorrelation(data, max_lag=40, optimal_delay=10)
        assert isinstance(ax, plt.Axes)
        # Should have legend from delay marker
        assert ax.get_legend() is not None


class TestDiagnosticsSummary:
    def test_returns_axes_array(self):
        data = _lorenz_x(2000)
        axes = plot_diagnostics_summary(data, window_size=200, step=100)
        assert len(axes) == 2
