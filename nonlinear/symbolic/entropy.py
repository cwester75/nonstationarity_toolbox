import numpy as np


def shannon_entropy(seq):
    vals = set(seq)

    probs = []

    for v in vals:
        probs.append(seq.count(v) / len(seq))

    return -sum(p * np.log2(p) for p in probs)
