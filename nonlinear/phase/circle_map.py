import numpy as np


class CircleMap:
    """Arnold tongue circle map: theta_{n+1} = theta_n + omega - (K/2pi)*sin(2pi*theta_n)."""

    def __init__(self, omega, K):
        self.omega = omega
        self.K = K

    def f(self, theta):
        return (theta + self.omega - (self.K / (2 * np.pi)) * np.sin(2 * np.pi * theta)) % 1.0

    def df(self, theta):
        return 1 - self.K * np.cos(2 * np.pi * theta)


def iterate_circle_map(omega, K, theta0, n):
    """Iterate the circle map and return the trajectory."""
    cm = CircleMap(omega, K)
    traj = [theta0]
    theta = theta0
    for _ in range(n):
        theta = cm.f(theta)
        traj.append(theta)
    return traj


def arnold_tongue(K_values, omega_values, theta0=0.1, n=500, tol=1e-4):
    """Compute Arnold tongue: identify mode-locked regions."""
    locked = []
    for K in K_values:
        for omega in omega_values:
            cm = CircleMap(omega, K)
            theta = theta0
            for _ in range(n):
                theta = cm.f(theta)

            # Check for period-1 locking
            theta_next = cm.f(theta)
            if abs(theta_next - theta) < tol:
                locked.append((omega, K))

    return locked
