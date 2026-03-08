"""Experiment: estimate Feigenbaum's delta constant from logistic map."""
from nonlinear.bifurcation.feigenbaum import (
    find_period_doubling_points,
    estimate_feigenbaum_delta,
    logistic_map_func,
)

print("Scanning for period-doubling bifurcation points...")
doublings = find_period_doubling_points(logistic_map_func, (2.5, 4.0))

print(f"Found {len(doublings)} period-doubling points:")
for i, r in enumerate(doublings):
    print(f"  r_{i+1} = {r:.6f}")

deltas = estimate_feigenbaum_delta(doublings)
if deltas:
    print(f"\nFeigenbaum delta estimates: {[f'{d:.4f}' for d in deltas]}")
    print(f"Theoretical value: 4.6692...")
