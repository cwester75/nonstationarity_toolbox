import numpy as np
from nonlinear.diagnostics.lyapunov import lyapunov_exponent


def is_chaotic(map_obj, x0=0.5, n=2000, threshold=0.01):
    """Determine if a map is chaotic based on Lyapunov exponent."""
    from nonlinear.core.trajectory import iterate_map
    traj = iterate_map(map_obj, x0, n)
    le = lyapunov_exponent(map_obj, traj)
    return le > threshold, le


def chaos_boundary(map_class, r_range, grid=500, x0=0.5, n=2000):
    """Scan parameter space for chaos onset."""
    r_values = np.linspace(r_range[0], r_range[1], grid)
    results = []

    for r in r_values:
        m = map_class(r)
        chaotic, le = is_chaotic(m, x0, n)
        results.append((r, le, chaotic))

    return results


def intermittency_detector(traj, window=50, threshold=0.1):
    """Detect intermittent bursts in a trajectory."""
    traj = np.array(traj)
    variances = []
    for i in range(0, len(traj) - window, window):
        segment = traj[i:i + window]
        variances.append(np.var(segment))

    variances = np.array(variances)
    mean_var = np.mean(variances)
    bursts = np.where(variances > mean_var * (1 + threshold))[0]
    return bursts, variances
