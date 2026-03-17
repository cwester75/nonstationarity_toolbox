from .core.maps import LogisticMap, TentMap
from .core.map2d import Map2D
from .core.trajectory import iterate_map
from .diagnostics.lyapunov import lyapunov_exponent
from .diagnostics.lyapunov_spectrum import lyapunov_spectrum
from .maps2d.henon_map import HenonMap
from .maps2d.baker_map import BakerMap
from .dynamical_systems import (
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
    detect_homoclinic,
    detect_limit_cycle,
)
from .hamiltonian import (
    HamiltonianSystem,
    integrate_hamiltonian,
    leapfrog_step,
    stormer_verlet_step,
    compute_poincare_section,
    compute_action_variable,
    check_energy_conservation,
    hamiltonian_lyapunov,
    classify_poincare_dynamics,
    harmonic_oscillator,
    henon_heiles,
)
