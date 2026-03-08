"""Experiment: generate Newton fractal for z^3 - 1 = 0."""
from nonlinear.optimization.newton_complex import newton_fractal_grid

# z^3 - 1
f = lambda z: z**3 - 1
df = lambda z: 3*z**2

print("Computing Newton fractal for z^3 - 1 = 0...")
root_grid, iter_grid, roots = newton_fractal_grid(
    f, df, (-2, 2), (-2, 2), grid=200, max_iter=30
)

print(f"Found {len(roots)} roots:")
for i, r in enumerate(roots):
    print(f"  root {i+1}: {r.real:.6f} + {r.imag:.6f}i")

print(f"\nGrid size: {root_grid.shape}")
print(f"Max iterations needed: {iter_grid.max()}")
print(f"Mean iterations: {iter_grid.mean():.1f}")
