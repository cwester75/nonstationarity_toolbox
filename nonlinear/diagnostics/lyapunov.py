import numpy as np


def lyapunov_exponent(map_obj, traj):
    """Compute the Lyapunov exponent from a trajectory of a 1-D map.

    Skips points where |df| = 0 to avoid log(0) warnings.
    Returns -inf for a superstable orbit (all derivatives zero).
    Raises ValueError on empty trajectory.
    """
    if len(traj) == 0:
        raise ValueError("Cannot compute Lyapunov exponent from empty trajectory")

    vals = []
    for x in traj:
        df_abs = abs(map_obj.df(x))
        if df_abs > 0:
            vals.append(np.log(df_abs))

    if len(vals) == 0:
        return float('-inf')

    return np.mean(vals)
