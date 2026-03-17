from .systems.base_system import AutonomousSystem2D
from .systems.pendulum import Pendulum, DampedPendulum
from .systems.lotka_volterra import LotkaVolterra
from .systems.biochemical import BiochemicalSystem
from .simulation.integrator import RK4Integrator, rk4_step, simulate
from .analysis.fixed_points import find_fixed_point, find_fixed_points, jacobian
from .analysis.stability import classify_fixed_point
from .analysis.homoclinic import detect_homoclinic
from .analysis.limit_cycles import detect_limit_cycle
from .analysis.hopf import detect_hopf_bifurcation, hopf_normal_form_coefficient
from .analysis.lyapunov_flow import (
    flow_lyapunov_exponent,
    flow_lyapunov_spectrum,
    flow_lyapunov_convergence,
)
from .analysis.attractors import classify_attractor, phase_space_contraction
from .analysis.dissipative_pipeline import dissipative_analysis
