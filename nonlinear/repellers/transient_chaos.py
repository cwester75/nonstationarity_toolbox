import numpy as np


def transient_lifetime(map_func, x0, escape_bound=1e6, max_iter=10000):
    """Measure how long a trajectory stays in the chaotic region before escaping."""
    x = x0
    for i in range(max_iter):
        x = map_func(x)
        if abs(x) > escape_bound:
            return i
    return max_iter


def survival_probability(map_func, x_range, n_samples=1000,
                         escape_bound=1e6, max_iter=5000):
    """Compute survival probability as a function of time."""
    x0s = np.random.uniform(x_range[0], x_range[1], n_samples)
    lifetimes = []
    for x0 in x0s:
        lifetimes.append(transient_lifetime(map_func, x0, escape_bound, max_iter))

    lifetimes = np.array(lifetimes)
    times = np.arange(max_iter)
    survival = np.array([np.sum(lifetimes > t) / n_samples for t in times])
    return times, survival


def escape_rate(times, survival):
    """Estimate escape rate kappa from survival probability P(t) ~ exp(-kappa*t)."""
    mask = survival > 0
    if np.sum(mask) < 2:
        return None
    t = times[mask]
    log_s = np.log(survival[mask])
    coeffs = np.polyfit(t, log_s, 1)
    return -coeffs[0]
