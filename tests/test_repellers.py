import numpy as np
from nonlinear.repellers.escape_map import escape_time, escape_time_grid, repeller_set


def test_escape_time_bounded():
    # Identity map never escapes
    t = escape_time(lambda x: x, 0.5, max_iter=100)
    assert t == 100


def test_escape_time_diverging():
    # x -> 2x escapes quickly from x=1
    t = escape_time(lambda x: 2 * x, 1.0, max_iter=1000, bound=100)
    assert t < 10


def test_escape_time_grid_shape():
    xs, times = escape_time_grid(lambda x: x, (0, 1), grid=50, max_iter=10)
    assert len(xs) == 50
    assert len(times) == 50


def test_repeller_set_nonempty():
    # Logistic map at r=4 on [0,1] never escapes
    repeller = repeller_set(lambda x: 4 * x * (1 - x), (0, 1),
                            grid=100, max_iter=200, bound=10)
    assert len(repeller) > 0
