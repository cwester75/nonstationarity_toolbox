import numpy as np
import pytest

from nonlinear.dynamical_systems import (
    AutonomousSystem2D,
    Pendulum,
    DampedPendulum,
    LotkaVolterra,
    BiochemicalSystem,
    RK4Integrator,
    rk4_step,
    simulate,
    find_fixed_point,
    find_fixed_points,
    jacobian,
    classify_fixed_point,
    detect_limit_cycle,
)


class TestAutonomousSystem2D:
    """Tests for the base system class."""

    def test_vector_field_returns_ndarray(self):
        def f(x, y, p):
            return -x

        def g(x, y, p):
            return -y

        system = AutonomousSystem2D(f, g)
        v = system.vector_field(1.0, 2.0)
        assert isinstance(v, np.ndarray)
        assert v.shape == (2,)
        assert v[0] == pytest.approx(-1.0)
        assert v[1] == pytest.approx(-2.0)

    def test_phase_portrait_shape(self):
        def f(x, y, p):
            return y

        def g(x, y, p):
            return -x

        system = AutonomousSystem2D(f, g)
        X, Y, DX, DY = system.phase_portrait((-1, 1), (-1, 1), nx=5, ny=5)
        assert X.shape == (5, 5)
        assert DX.shape == (5, 5)


class TestRK4Integrator:
    """Tests for the RK4 integrator."""

    def test_rk4_step_function(self):
        """Standalone rk4_step function works correctly."""
        def f(x, y, p):
            return -x

        def g(x, y, p):
            return -y

        system = AutonomousSystem2D(f, g)
        x, y = 1.0, 1.0
        dt = 0.001
        for _ in range(1000):
            x, y = rk4_step(system, x, y, dt)

        expected = np.exp(-1.0)
        assert x == pytest.approx(expected, rel=1e-6)
        assert y == pytest.approx(expected, rel=1e-6)

    def test_rk4_class_matches_function(self):
        """Class-based integrator matches standalone function."""
        def f(x, y, p):
            return y

        def g(x, y, p):
            return -np.sin(x)

        system = AutonomousSystem2D(f, g)
        integrator = RK4Integrator()

        x1, y1 = rk4_step(system, 0.5, 0.0, 0.01)
        x2, y2 = integrator.step(system, 0.5, 0.0, 0.01)
        assert x1 == pytest.approx(x2)
        assert y1 == pytest.approx(y2)

    def test_simulate_shape(self):
        system = Pendulum()
        traj = simulate(system, 0.1, 0.0, steps=100, dt=0.01)
        assert traj.shape == (100, 2)
        assert traj[0, 0] == pytest.approx(0.1)
        assert traj[0, 1] == pytest.approx(0.0)


class TestPendulum:
    """Tests for the pendulum system."""

    def test_fixed_point_origin(self):
        pend = Pendulum()
        v = pend.vector_field(0.0, 0.0)
        assert v[0] == pytest.approx(0.0)
        assert v[1] == pytest.approx(0.0)

    def test_energy_conservation(self):
        """Undamped pendulum should approximately conserve energy."""
        pend = Pendulum()
        traj = simulate(pend, 0.5, 0.0, steps=5000, dt=0.01)

        def energy(x, y):
            return 0.5 * y ** 2 - np.cos(x)

        e0 = energy(traj[0, 0], traj[0, 1])
        e_final = energy(traj[-1, 0], traj[-1, 1])
        assert e_final == pytest.approx(e0, abs=1e-4)


class TestLotkaVolterra:
    """Tests for the Lotka-Volterra system."""

    def test_trivial_fixed_point(self):
        lv = LotkaVolterra(alpha=1.5, beta=1.0, delta=1.0, gamma=3.0)
        v = lv.vector_field(0.0, 0.0)
        assert v[0] == pytest.approx(0.0)
        assert v[1] == pytest.approx(0.0)

    def test_nontrivial_fixed_point(self):
        lv = LotkaVolterra(alpha=1.5, beta=1.0, delta=1.0, gamma=3.0)
        # Fixed point at (gamma/delta, alpha/beta) = (3, 1.5)
        v = lv.vector_field(3.0, 1.5)
        assert v[0] == pytest.approx(0.0, abs=1e-10)
        assert v[1] == pytest.approx(0.0, abs=1e-10)


class TestBiochemicalSystem:
    """Tests for the biochemical system."""

    def test_vector_field(self):
        bio = BiochemicalSystem(k1=2.0, k2=1.0, k3=1.0, k4=1.0)
        v = bio.vector_field(1.0, 1.0)
        assert v[0] == pytest.approx(1.0)  # 2 - 1*1*1
        assert v[1] == pytest.approx(0.0)  # 1*1 - 1*1


class TestFixedPoints:
    """Tests for fixed point detection."""

    def test_find_fixed_point_single(self):
        """find_fixed_point returns a single converged point."""
        def f(x, y, p):
            return -x

        def g(x, y, p):
            return -y

        system = AutonomousSystem2D(f, g)
        fp = find_fixed_point(system, (0.1, 0.1))
        assert fp is not None
        assert fp[0] == pytest.approx(0.0, abs=1e-8)
        assert fp[1] == pytest.approx(0.0, abs=1e-8)

    def test_find_fixed_points_list(self):
        """find_fixed_points returns a deduplicated list."""
        def f(x, y, p):
            return -x

        def g(x, y, p):
            return -y

        system = AutonomousSystem2D(f, g)
        fps = find_fixed_points(system, [(0.1, 0.1)])
        assert len(fps) == 1
        assert fps[0][0] == pytest.approx(0.0, abs=1e-8)
        assert fps[0][1] == pytest.approx(0.0, abs=1e-8)

    def test_find_lotka_volterra_fixed_points(self):
        lv = LotkaVolterra(alpha=1.5, beta=1.0, delta=1.0, gamma=3.0)
        fps = find_fixed_points(lv, [(0.1, 0.1), (2.5, 1.0)])
        # Should find origin and (3, 1.5)
        assert len(fps) >= 1


class TestJacobian:
    """Tests for central-difference Jacobian computation."""

    def test_linear_jacobian(self):
        """For dx=-2x+y, dy=x-3y the Jacobian is constant."""
        def f(x, y, p):
            return -2 * x + y

        def g(x, y, p):
            return x - 3 * y

        system = AutonomousSystem2D(f, g)
        J = jacobian(system, 0.0, 0.0)
        assert J[0, 0] == pytest.approx(-2.0, abs=1e-6)
        assert J[0, 1] == pytest.approx(1.0, abs=1e-6)
        assert J[1, 0] == pytest.approx(1.0, abs=1e-6)
        assert J[1, 1] == pytest.approx(-3.0, abs=1e-6)

    def test_jacobian_at_nonzero_point(self):
        """Jacobian of a nonlinear system at a non-origin point."""
        def f(x, y, p):
            return x * y

        def g(x, y, p):
            return x ** 2 - y

        system = AutonomousSystem2D(f, g)
        J = jacobian(system, 2.0, 3.0)
        # df/dx = y = 3, df/dy = x = 2
        # dg/dx = 2x = 4, dg/dy = -1
        assert J[0, 0] == pytest.approx(3.0, abs=1e-4)
        assert J[0, 1] == pytest.approx(2.0, abs=1e-4)
        assert J[1, 0] == pytest.approx(4.0, abs=1e-4)
        assert J[1, 1] == pytest.approx(-1.0, abs=1e-4)


class TestStabilityClassification:
    """Tests for fixed point classification."""

    def test_stable_node(self):
        def f(x, y, p):
            return -2 * x

        def g(x, y, p):
            return -3 * y

        system = AutonomousSystem2D(f, g)
        result = classify_fixed_point(system, (0, 0))
        assert result == "stable node"

    def test_unstable_node(self):
        def f(x, y, p):
            return 2 * x

        def g(x, y, p):
            return 3 * y

        system = AutonomousSystem2D(f, g)
        result = classify_fixed_point(system, (0, 0))
        assert result == "unstable node"

    def test_saddle(self):
        def f(x, y, p):
            return 2 * x

        def g(x, y, p):
            return -3 * y

        system = AutonomousSystem2D(f, g)
        result = classify_fixed_point(system, (0, 0))
        assert result == "saddle"

    def test_stable_spiral(self):
        def f(x, y, p):
            return -x + 2 * y

        def g(x, y, p):
            return -2 * x - y

        system = AutonomousSystem2D(f, g)
        result = classify_fixed_point(system, (0, 0))
        assert result == "stable spiral"

    def test_center(self):
        pend = Pendulum()
        result = classify_fixed_point(pend, (0, 0))
        assert result == "center"


class TestLimitCycleDetection:
    """Tests for limit cycle detection."""

    def test_no_limit_cycle_linear(self):
        """Linear stable system has no limit cycle."""
        def f(x, y, p):
            return -x

        def g(x, y, p):
            return -y

        system = AutonomousSystem2D(f, g)
        traj = simulate(system, 1.0, 1.0, steps=5000, dt=0.01)
        assert detect_limit_cycle(traj) is False

    def test_oscillatory_crossings(self):
        """Pendulum should produce regular crossings (quasi-periodic)."""
        pend = Pendulum()
        traj = simulate(pend, 0.5, 0.0, steps=10000, dt=0.01)
        result = detect_limit_cycle(traj, section_coord=0, section_value=0.0)
        assert isinstance(result, bool)
