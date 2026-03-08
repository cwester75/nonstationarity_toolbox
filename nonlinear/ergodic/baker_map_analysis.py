import numpy as np
from nonlinear.maps2d.baker_map import BakerMap


def ergodicity_test(map2d, state0, observable, n=10000, transient=1000):
    """Test ergodicity by comparing time average to space average.

    If ergodic, time average of observable along orbit = space average.
    """
    state = np.array(state0, dtype=float)

    # Discard transient
    for _ in range(transient):
        state = np.array(map2d.step(state), dtype=float)

    # Time average
    time_sum = 0.0
    for _ in range(n):
        state = np.array(map2d.step(state), dtype=float)
        time_sum += observable(state)
    time_avg = time_sum / n

    # Space average (uniform on unit square for baker map)
    n_space = 5000
    space_sum = 0.0
    for _ in range(n_space):
        pt = np.random.uniform(0, 1, 2)
        space_sum += observable(pt)
    space_avg = space_sum / n_space

    return time_avg, space_avg, abs(time_avg - space_avg)


def mixing_test(map2d, set_A_func, set_B_func, n_iter=50, n_points=5000):
    """Test mixing by checking if measure(A ∩ T^{-n}(B)) -> mu(A)*mu(B).

    set_A_func, set_B_func: indicator functions returning True if point is in set.
    """
    # Estimate mu(A) and mu(B)
    points = np.random.uniform(0, 1, (n_points, 2))
    mu_A = np.mean([set_A_func(p) for p in points])
    mu_B = np.mean([set_B_func(p) for p in points])

    correlations = []
    for n in range(1, n_iter + 1):
        count = 0
        total = 0
        for p in points:
            if set_A_func(p):
                # Iterate n times
                state = p.copy()
                for _ in range(n):
                    state = np.array(map2d.step(state), dtype=float)
                if set_B_func(state):
                    count += 1
                total += 1

        if total > 0:
            correlations.append(count / total)
        else:
            correlations.append(0)

    return correlations, mu_A, mu_B
