import numpy as np


def uniform_partition(x, n_symbols):
    """Partition [0,1] into n_symbols equal bins."""
    return min(int(x * n_symbols), n_symbols - 1)


def encode_with_partition(traj, n_symbols=2):
    """Encode trajectory using uniform partition."""
    return [uniform_partition(x, n_symbols) for x in traj]


def word_frequencies(seq, word_length):
    """Compute frequencies of all words of given length."""
    freqs = {}
    for i in range(len(seq) - word_length + 1):
        word = tuple(seq[i:i + word_length])
        freqs[word] = freqs.get(word, 0) + 1

    total = sum(freqs.values())
    return {k: v / total for k, v in freqs.items()}


def block_entropy(seq, word_length):
    """Compute block entropy H_n for words of given length."""
    freqs = word_frequencies(seq, word_length)
    return -sum(p * np.log2(p) for p in freqs.values() if p > 0)


def topological_entropy_estimate(seq, max_word_length=10):
    """Estimate topological entropy from growth rate of distinct words."""
    lengths = range(1, max_word_length + 1)
    n_words = []

    for L in lengths:
        words = set()
        for i in range(len(seq) - L + 1):
            words.add(tuple(seq[i:i + L]))
        n_words.append(len(words))

    # h_top ≈ lim log(N_L) / L
    log_n = np.log(np.array(n_words, dtype=float))
    L_arr = np.array(list(lengths), dtype=float)
    coeffs = np.polyfit(L_arr, log_n, 1)
    return coeffs[0]
