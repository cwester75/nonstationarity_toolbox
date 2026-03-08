import numpy as np


def find_period_doubling_points(map_func, r_range, x0=0.5, iterations=2000,
                                 settle=1500, tol=1e-6):
    """Find parameter values where period doublings occur."""
    r_values = np.linspace(r_range[0], r_range[1], 50000)
    doublings = []
    prev_period = 1

    for r in r_values:
        x = x0
        for _ in range(settle):
            x = map_func(r, x)

        orbit = []
        for _ in range(iterations - settle):
            x = map_func(r, x)
            orbit.append(x)

        unique = [orbit[0]]
        for v in orbit[1:]:
            is_new = True
            for u in unique:
                if abs(v - u) < tol:
                    is_new = False
                    break
            if is_new:
                unique.append(v)

        period = len(unique)
        if period == 2 * prev_period and period <= 64:
            doublings.append(r)
            prev_period = period

    return doublings


def estimate_feigenbaum_delta(doublings):
    """Estimate Feigenbaum delta from a sequence of period-doubling points."""
    if len(doublings) < 3:
        return None

    deltas = []
    for i in range(len(doublings) - 2):
        num = doublings[i + 1] - doublings[i]
        den = doublings[i + 2] - doublings[i + 1]
        if abs(den) > 1e-15:
            deltas.append(num / den)

    return deltas


def logistic_map_func(r, x):
    """Logistic map as a simple function for bifurcation scanning."""
    return r * x * (1 - x)
