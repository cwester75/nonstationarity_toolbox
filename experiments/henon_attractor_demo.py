"""Experiment: compute and analyze the Henon attractor."""
import numpy as np
from nonlinear.maps2d.henon_map import HenonMap
from nonlinear.diagnostics.lyapunov_spectrum import lyapunov_spectrum, kaplan_yorke_dimension

henon = HenonMap(a=1.4, b=0.3)

# Fixed points
fps = henon.fixed_points()
print("Fixed points:")
for fp in fps:
    print(f"  ({fp[0]:.6f}, {fp[1]:.6f})")

# Lyapunov spectrum
exponents = lyapunov_spectrum(henon, (0.1, 0.1), n=10000)
print(f"\nLyapunov exponents: {exponents[0]:.4f}, {exponents[1]:.4f}")
print(f"Sum: {sum(exponents):.4f}")

d_ky = kaplan_yorke_dimension(exponents)
print(f"Kaplan-Yorke dimension: {d_ky:.4f}")

# Generate attractor
traj = henon.iterate((0.1, 0.1), 50000)
traj = traj[1000:]  # discard transient
print(f"\nAttractor bounding box:")
print(f"  x: [{traj[:, 0].min():.4f}, {traj[:, 0].max():.4f}]")
print(f"  y: [{traj[:, 1].min():.4f}, {traj[:, 1].max():.4f}]")
