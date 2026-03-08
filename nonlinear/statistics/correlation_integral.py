import numpy as np


def correlation_integral(data, epsilon):
    """Compute the Grassberger-Procaccia correlation integral C(epsilon).

    data: array of shape (N, d) or (N,) for 1-D.
    """
    data = np.atleast_2d(data)
    if data.shape[0] == 1:
        data = data.T
    N = data.shape[0]

    count = 0
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.linalg.norm(data[i] - data[j])
            if dist < epsilon:
                count += 1

    return 2.0 * count / (N * (N - 1))


def correlation_dimension(data, epsilons=None, n_eps=20):
    """Estimate correlation dimension from the scaling of C(epsilon).

    Returns (epsilons, C_values, estimated_dimension).
    """
    data = np.atleast_2d(data)
    if data.shape[0] == 1:
        data = data.T

    if epsilons is None:
        dists = []
        N = min(data.shape[0], 500)
        for i in range(N):
            for j in range(i + 1, N):
                dists.append(np.linalg.norm(data[i] - data[j]))
        dists = np.array(dists)
        eps_min = np.percentile(dists, 1)
        eps_max = np.percentile(dists, 50)
        if eps_min <= 0:
            eps_min = 1e-10
        epsilons = np.logspace(np.log10(eps_min), np.log10(eps_max), n_eps)

    C_values = []
    for eps in epsilons:
        C_values.append(correlation_integral(data, eps))

    C_values = np.array(C_values)

    # Estimate dimension from log-log slope
    mask = C_values > 0
    if np.sum(mask) < 2:
        return epsilons, C_values, None

    log_eps = np.log(epsilons[mask])
    log_C = np.log(C_values[mask])
    coeffs = np.polyfit(log_eps, log_C, 1)

    return epsilons, C_values, coeffs[0]
