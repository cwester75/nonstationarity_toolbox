from nonlinear.core.maps import LogisticMap


def test_logistic():
    m = LogisticMap(3.5)

    assert abs(m.f(0.5) - 0.875) < 1e-6
