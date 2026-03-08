import numpy as np
from nonlinear.symbolic.partition import block_entropy, encode_with_partition
from nonlinear.symbolic.entropy import shannon_entropy


def test_block_entropy_random():
    """Block entropy of a random binary sequence should be close to 1."""
    np.random.seed(42)
    seq = list(np.random.randint(0, 2, 10000))
    h = shannon_entropy(seq)
    assert abs(h - 1.0) < 0.05


def test_uniform_partition():
    """Test that uniform partition produces expected symbols."""
    seq = encode_with_partition([0.1, 0.6, 0.3, 0.9], n_symbols=2)
    assert seq == [0, 1, 0, 1]
