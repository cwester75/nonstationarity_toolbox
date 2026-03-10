"""Chapter 3 demonstration: Lotka-Volterra phase-space analysis.

Simulates a predator-prey system, locates fixed points, classifies
their stability, and checks for limit cycles.

Usage:
    python -m nonlinear.dynamical_systems.examples.run_demo
"""
from nonlinear.dynamical_systems import (
    LotkaVolterra,
    simulate,
    find_fixed_point,
    classify_fixed_point,
    detect_limit_cycle,
)


def main():
    system = LotkaVolterra()

    traj = simulate(system, 2, 1, steps=10000)

    fp = find_fixed_point(system, (1, 1))

    print("Fixed point:", fp)
    print("Type:", classify_fixed_point(system, fp))

    cycle = detect_limit_cycle(traj)
    print("Limit cycle detected:", cycle)

    # Optional matplotlib visualisation
    try:
        import matplotlib.pyplot as plt

        plt.plot(traj[:, 0], traj[:, 1])
        plt.xlabel("x (prey)")
        plt.ylabel("y (predator)")
        plt.title("Lotka-Volterra phase trajectory")
        plt.show()
    except ImportError:
        print("(matplotlib not available, skipping plot)")


if __name__ == "__main__":
    main()
