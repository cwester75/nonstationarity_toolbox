"""Henon attractor demo — phase portrait, fixed points, Lyapunov spectrum."""
import numpy as np
from nonlinear.maps2d.henon_map import HenonMap
from nonlinear.maps2d.map2d import iterate_map
from nonlinear.diagnostics.fixed_points_2d import find_fixed_point, stability
from nonlinear.diagnostics.lyapunov_spectrum import lyapunov_spectrum, kaplan_yorke_dimension

# Create Henon map with classic parameters
m = HenonMap()

# Iterate
xs, ys = iterate_map(m, 0.1, 0.1, 10000)

# Fixed points
fp1 = find_fixed_point(m, [0.5, 0.2])
fp2 = find_fixed_point(m, [-1.0, -0.3])

print("Fixed points:")
print(f"  fp1 = ({fp1[0]:.6f}, {fp1[1]:.6f})")
print(f"  fp2 = ({fp2[0]:.6f}, {fp2[1]:.6f})")

# Stability
eig1 = stability(m, fp1)
eig2 = stability(m, fp2)
print(f"\nEigenvalues at fp1: {eig1}")
print(f"Eigenvalues at fp2: {eig2}")

# Lyapunov spectrum
lam = lyapunov_spectrum(m, (0.1, 0.1), n=10000)
print(f"\nLyapunov spectrum: λ1={lam[0]:.4f}, λ2={lam[1]:.4f}")
print(f"Kaplan-Yorke dimension: {kaplan_yorke_dimension(lam):.4f}")

# Phase portrait (only show if display available)
try:
    from nonlinear.visualization.phase_portrait import plot_phase_portrait
    plot_phase_portrait(xs, ys)
except Exception:
    print(f"\nAttractor bounds: x=[{xs.min():.3f}, {xs.max():.3f}], y=[{ys.min():.3f}, {ys.max():.3f}]")
