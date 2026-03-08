import numpy as np


def find_fixed_points(map_obj, grid=1000):
    xs = np.linspace(0, 1, grid)

    pts = []

    for x in xs:
        if abs(map_obj.f(x) - x) < 1e-4:
            pts.append(x)

    return pts
