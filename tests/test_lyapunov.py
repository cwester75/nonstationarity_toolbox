from nonlinear.core.maps import LogisticMap
from nonlinear.core.trajectory import iterate_map
from nonlinear.diagnostics.lyapunov import lyapunov_exponent


def test_positive():
    m = LogisticMap(4)

    traj = iterate_map(m, 0.2, 1000)

    assert lyapunov_exponent(m, traj) > 0
