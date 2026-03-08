import numpy as np
from nonlinear.statistics.autocorrelation import acf
from nonlinear.statistics.invariant_density import estimate_density, logistic_invariant_density
from nonlinear.statistics.random_maps import bernoulli_map, generate_random_sequence, uniformity_test
from nonlinear.statistics.correlation_integral import correlation_integral
from nonlinear.statistics.capacity_dimension import box_count_1d, box_count_2d, capacity_dimension


def test_acf_lag1():
    # Sine wave should have positive lag-1 autocorrelation
    t = np.linspace(0, 10 * np.pi, 1000)
    s = np.sin(t)
    assert acf(s, lag=1) > 0.9


def test_acf_edge_cases():
    assert acf([1, 2, 3], lag=0) == 0.0  # lag=0 returns 0
    assert acf([1, 2, 3], lag=5) == 0.0  # lag >= len returns 0
    assert acf([5, 5, 5], lag=1) == 0.0  # zero variance


def test_estimate_density_sums_to_one():
    np.random.seed(42)
    data = np.random.rand(10000)
    centers, counts = estimate_density(data, bins=100)
    dx = centers[1] - centers[0]
    total = np.sum(counts) * dx
    assert abs(total - 1.0) < 0.1


def test_logistic_invariant_density_boundaries():
    # Should be 0 outside (0, 1) and finite inside
    assert logistic_invariant_density(0.5) > 0
    assert logistic_invariant_density(-0.1) == 0.0
    assert logistic_invariant_density(1.1) == 0.0


def test_bernoulli_map_range():
    x = 0.3
    for _ in range(100):
        x = bernoulli_map(x)
        assert 0 <= x < 1


def test_uniformity_test():
    np.random.seed(42)
    seq = list(np.random.rand(5000))
    chi2 = uniformity_test(seq, bins=50)
    # Uniform data should have low chi2
    assert chi2 < 200


def test_correlation_integral_increases():
    np.random.seed(42)
    data = np.random.rand(100, 2)
    c1 = correlation_integral(data, 0.1)
    c2 = correlation_integral(data, 0.5)
    assert c2 >= c1


def test_box_count_1d_increases():
    data = np.random.rand(500)
    sizes, counts = box_count_1d(data)
    # Smaller boxes should give more counts
    assert counts[0] >= counts[-1]


def test_box_count_2d():
    data = np.random.rand(200, 2)
    sizes, counts = box_count_2d(data)
    assert len(sizes) == len(counts)
    assert all(c > 0 for c in counts)


def test_capacity_dimension_returns_float():
    sizes = np.array([0.1, 0.05, 0.01])
    counts = np.array([10, 30, 200])
    d = capacity_dimension(sizes, counts)
    assert isinstance(d, float)
    assert d > 0
