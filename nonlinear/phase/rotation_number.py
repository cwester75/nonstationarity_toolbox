import numpy as np


def rotation_number(omega, K, theta0=0.0, n=5000):
    """Compute the rotation number of a circle map."""
    theta = theta0
    total = 0.0

    for _ in range(n):
        theta_new = theta + omega - (K / (2 * np.pi)) * np.sin(2 * np.pi * theta)
        total += theta_new - theta
        theta = theta_new % 1.0

    return total / n


def rotation_number_scan(K, omega_values, theta0=0.0, n=5000):
    """Compute rotation number as a function of omega (devil's staircase)."""
    rots = []
    for omega in omega_values:
        rots.append(rotation_number(omega, K, theta0, n))
    return np.array(rots)
