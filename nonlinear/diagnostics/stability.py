import numpy as np


def classify_fixed_point_1d(map_obj, x_fixed, tol=1e-8):
    """Classify a 1-D fixed point by the derivative magnitude."""
    df = abs(map_obj.df(x_fixed))

    if df < 1 - tol:
        return "stable"
    elif df > 1 + tol:
        return "unstable"
    else:
        return "marginal"


def stability_multiplier(map_obj, orbit):
    """Compute the stability multiplier of a periodic orbit."""
    product = 1.0
    for x in orbit:
        product *= map_obj.df(x)
    return product


def is_superstable(map_obj, orbit, tol=1e-8):
    """Check if a periodic orbit is superstable (multiplier = 0)."""
    return abs(stability_multiplier(map_obj, orbit)) < tol
