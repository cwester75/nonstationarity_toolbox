import numpy as np


def lyapunov_exponent(map_obj, traj):
    vals = []

    for x in traj:
        vals.append(np.log(abs(map_obj.df(x))))

    return np.mean(vals)
