import numpy as np
from scipy import integrate


def melnikov_integral(q0_func, dq0_func, perturbation_func, t_span=(-50, 50), t0=0):
    """Compute the Melnikov integral for detecting homoclinic chaos.

    M(t0) = integral from -inf to +inf of
             dq0/dt(t - t0) * perturbation(q0(t - t0), t) dt

    q0_func: unperturbed homoclinic orbit q0(t)
    dq0_func: derivative of homoclinic orbit dq0/dt(t)
    perturbation_func: perturbation g(q, t)
    """
    def integrand(t):
        tau = t - t0
        q0 = q0_func(tau)
        dq0 = dq0_func(tau)
        return dq0 * perturbation_func(q0, t)

    result, _ = integrate.quad(integrand, t_span[0], t_span[1])
    return result


def melnikov_scan(q0_func, dq0_func, perturbation_func, t0_values,
                  t_span=(-50, 50)):
    """Compute Melnikov function M(t0) for a range of t0 values.

    If M(t0) changes sign, homoclinic intersection exists => chaos.
    """
    M_values = []
    for t0 in t0_values:
        M = melnikov_integral(q0_func, dq0_func, perturbation_func, t_span, t0)
        M_values.append(M)
    return np.array(M_values)


def has_simple_zeros(M_values):
    """Check if the Melnikov function has simple zeros (sign changes)."""
    sign_changes = 0
    for i in range(len(M_values) - 1):
        if M_values[i] * M_values[i + 1] < 0:
            sign_changes += 1
    return sign_changes > 0, sign_changes
