import numpy as np


def lyapunov_spectrum(map2d, state0, n=5000, transient=500):
    """Compute the Lyapunov spectrum of a 2-D map using QR decomposition.

    Returns (lambda1, lambda2) - the two Lyapunov exponents.
    """
    state = np.array(state0, dtype=float)
    dim = 2

    # Discard transient
    for _ in range(transient):
        state = np.array(map2d.step(state), dtype=float)

    Q = np.eye(dim)
    lyap_sums = np.zeros(dim)

    for _ in range(n):
        J = map2d.jacobian(state)
        state = np.array(map2d.step(state), dtype=float)

        # Evolve the tangent vectors
        M = J @ Q
        Q, R = np.linalg.qr(M)

        # Accumulate log of diagonal elements
        for j in range(dim):
            if abs(R[j, j]) > 0:
                lyap_sums[j] += np.log(abs(R[j, j]))

    return lyap_sums / n


def kaplan_yorke_dimension(exponents):
    """Compute the Kaplan-Yorke dimension from Lyapunov exponents."""
    sorted_exp = sorted(exponents, reverse=True)
    cumsum = 0.0

    for j, lam in enumerate(sorted_exp):
        cumsum += lam
        if cumsum < 0:
            # D_KY = j + sum(lambda_1..lambda_j) / |lambda_{j+1}|
            d_ky = j + (cumsum - lam) / abs(lam)
            return d_ky

    return len(exponents)
