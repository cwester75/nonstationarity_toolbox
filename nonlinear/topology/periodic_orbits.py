import numpy as np


def find_periodic_orbits_1d(map_obj, period, x_range=(0, 1), grid=1000, tol=1e-8):
    """Find periodic orbits of a given period for a 1-D map.

    Uses grid search + Newton refinement on f^p(x) - x = 0.
    """
    def f_composed(x, p):
        val = x
        for _ in range(p):
            val = map_obj.f(val)
        return val

    xs = np.linspace(x_range[0], x_range[1], grid)
    orbits = []

    for x0 in xs:
        # Newton's method on F(x) = f^p(x) - x
        x = x0
        for _ in range(50):
            fp = f_composed(x, period)
            residual = fp - x

            if abs(residual) < tol:
                # Verify it's truly period-p (not a divisor)
                is_lower_period = False
                for d in range(1, period):
                    if period % d == 0:
                        if abs(f_composed(x, d) - x) < tol:
                            is_lower_period = True
                            break

                if not is_lower_period:
                    is_new = True
                    for orb in orbits:
                        if any(abs(x - pt) < tol * 100 for pt in orb):
                            is_new = False
                            break
                    if is_new:
                        orbit = [x]
                        xi = x
                        for _ in range(period - 1):
                            xi = map_obj.f(xi)
                            orbit.append(xi)
                        orbits.append(orbit)
                break

            # Numerical derivative of f^p
            h = 1e-8
            dfp = (f_composed(x + h, period) - fp) / h
            if abs(dfp - 1) < 1e-12:
                break
            x = x - residual / (dfp - 1)

    return orbits


def topological_degree(map_obj, x_range=(0, 1), grid=10000):
    """Compute topological degree of a map on an interval.

    Counts the number of times f(x) - x crosses zero with sign.
    """
    xs = np.linspace(x_range[0], x_range[1], grid)
    degree = 0

    for i in range(grid - 1):
        g0 = map_obj.f(xs[i]) - xs[i]
        g1 = map_obj.f(xs[i + 1]) - xs[i + 1]
        if g0 * g1 < 0:
            if g1 > g0:
                degree += 1
            else:
                degree -= 1

    return degree
