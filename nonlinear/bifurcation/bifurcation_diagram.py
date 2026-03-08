import numpy as np


def generate_bifurcation(r_values, iterations=1000, last=100):
    rs = []
    xs = []

    for r in r_values:
        x = 0.5

        for i in range(iterations):
            x = r * x * (1 - x)

            if i >= iterations - last:
                rs.append(r)
                xs.append(x)

    return rs, xs
