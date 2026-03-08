import numpy as np


def box_count_1d(data, box_sizes=None, n_sizes=15):
    """Box-counting for 1-D data.

    Returns (box_sizes, counts).
    """
    data = np.asarray(data)
    data_range = data.max() - data.min()

    if box_sizes is None:
        box_sizes = np.logspace(
            np.log10(data_range / 1000),
            np.log10(data_range / 2),
            n_sizes
        )

    counts = []
    for eps in box_sizes:
        bins = np.arange(data.min(), data.max() + eps, eps)
        hist, _ = np.histogram(data, bins=bins)
        counts.append(np.sum(hist > 0))

    return np.array(box_sizes), np.array(counts)


def box_count_2d(data, box_sizes=None, n_sizes=15):
    """Box-counting for 2-D point set.

    data: array of shape (N, 2).
    Returns (box_sizes, counts).
    """
    data = np.asarray(data)
    x_range = data[:, 0].max() - data[:, 0].min()
    y_range = data[:, 1].max() - data[:, 1].min()
    max_range = max(x_range, y_range)

    if box_sizes is None:
        box_sizes = np.logspace(
            np.log10(max_range / 500),
            np.log10(max_range / 2),
            n_sizes
        )

    counts = []
    for eps in box_sizes:
        x_bins = np.floor((data[:, 0] - data[:, 0].min()) / eps).astype(int)
        y_bins = np.floor((data[:, 1] - data[:, 1].min()) / eps).astype(int)
        occupied = set(zip(x_bins, y_bins))
        counts.append(len(occupied))

    return np.array(box_sizes), np.array(counts)


def capacity_dimension(box_sizes, counts):
    """Estimate capacity (box-counting) dimension from log-log slope."""
    mask = counts > 0
    if np.sum(mask) < 2:
        return None

    log_eps = np.log(1.0 / box_sizes[mask])
    log_N = np.log(counts[mask].astype(float))
    coeffs = np.polyfit(log_eps, log_N, 1)
    return coeffs[0]
