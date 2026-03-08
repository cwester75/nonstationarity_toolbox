"""Experiment: observe period-doubling cascade in the logistic map."""
import numpy as np
from nonlinear.core.maps import LogisticMap
from nonlinear.core.trajectory import iterate_map
from nonlinear.core.orbit_tools import detect_periodicity

r_values = [2.5, 3.2, 3.5, 3.56, 3.565, 3.57, 4.0]

for r in r_values:
    m = LogisticMap(r)
    traj = iterate_map(m, 0.2, 5000)
    period = detect_periodicity(traj[-200:], tol=1e-4)
    print(f"r = {r:.3f}  period = {period}")
