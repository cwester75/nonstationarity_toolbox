import numpy as np


def bernoulli_map(x, a=2):
    """Bernoulli shift map: x -> (a*x) mod 1."""
    return (a * x) % 1


def tent_random(x, mu=2):
    """Tent map used as random number generator."""
    if x < 0.5:
        return mu * x
    return mu * (1 - x)


def generate_random_sequence(map_func, x0, n, **kwargs):
    """Generate pseudo-random sequence from a chaotic map."""
    seq = []
    x = x0
    for _ in range(n):
        x = map_func(x, **kwargs)
        seq.append(x)
    return seq


def uniformity_test(seq, bins=50):
    """Test uniformity of a sequence by comparing histogram to uniform."""
    counts, _ = np.histogram(seq, bins=bins, range=(0, 1))
    expected = len(seq) / bins
    chi2 = np.sum((counts - expected) ** 2 / expected)
    return chi2
