import numpy as np


def shannon_entropy(seq):
    """Compute Shannon entropy (in bits) of a discrete sequence.

    Returns 0.0 for empty sequences.
    """
    if len(seq) == 0:
        return 0.0

    vals = set(seq)
    n = len(seq)
    probs = [seq.count(v) / n for v in vals]

    return -sum(p * np.log2(p) for p in probs if p > 0)
