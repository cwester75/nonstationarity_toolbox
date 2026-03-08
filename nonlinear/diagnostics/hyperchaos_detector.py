import numpy as np
from nonlinear.diagnostics.lyapunov_spectrum import lyapunov_spectrum


def detect_hyperchaos(map2d, state0, n=5000, transient=500):
    """Detect hyperchaos: two or more positive Lyapunov exponents.

    Returns (is_hyperchaotic, exponents).
    """
    exponents = lyapunov_spectrum(map2d, state0, n, transient)
    positive_count = sum(1 for e in exponents if e > 0)
    return positive_count >= 2, exponents


def chaos_type(exponents, tol=0.01):
    """Classify dynamics from Lyapunov exponents."""
    pos = sum(1 for e in exponents if e > tol)
    zero = sum(1 for e in exponents if abs(e) <= tol)
    neg = sum(1 for e in exponents if e < -tol)

    if pos == 0 and zero == 0:
        return "stable fixed point"
    elif pos == 0 and zero >= 1:
        return "periodic or quasi-periodic"
    elif pos == 1:
        return "chaotic"
    elif pos >= 2:
        return "hyperchaotic"
    else:
        return "unknown"
