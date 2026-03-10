import numpy as np
import matplotlib.pyplot as plt


# ── recurrence plot ─────────────────────────────────────────────────

def plot_recurrence_matrix(R, ax=None, **kwargs):
    """Plot a recurrence matrix as a black-and-white image.

    Parameters
    ----------
    R : ndarray, shape (N, N)
        Boolean recurrence matrix (e.g. from
        ``RecurrenceAnalysis.recurrence_matrix``).
    ax : matplotlib Axes, optional
    **kwargs
        Forwarded to ``ax.imshow``.

    Returns
    -------
    matplotlib Axes
    """
    if ax is None:
        fig, ax = plt.subplots()

    kw = dict(cmap='Greys', origin='lower', interpolation='nearest',
              aspect='equal')
    kw.update(kwargs)
    ax.imshow(R.astype(float), **kw)
    ax.set_xlabel('i')
    ax.set_ylabel('j')
    ax.set_title('Recurrence Plot')
    return ax


def plot_recurrence_rate_vs_epsilon(recurrence_analysis, epsilons, ax=None):
    """Plot recurrence rate as a function of threshold epsilon.

    Parameters
    ----------
    recurrence_analysis : RecurrenceAnalysis
    epsilons : array-like
        Thresholds to scan.
    ax : matplotlib Axes, optional

    Returns
    -------
    matplotlib Axes
    """
    rates = [recurrence_analysis.recurrence_rate(e) for e in epsilons]

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(epsilons, rates, '.-')
    ax.set_xlabel('epsilon')
    ax.set_ylabel('Recurrence Rate')
    ax.set_title('Recurrence Rate vs Threshold')
    return ax


def plot_determinism_vs_epsilon(recurrence_analysis, epsilons, l_min=2,
                                ax=None):
    """Plot determinism as a function of threshold epsilon.

    Parameters
    ----------
    recurrence_analysis : RecurrenceAnalysis
    epsilons : array-like
    l_min : int
    ax : matplotlib Axes, optional

    Returns
    -------
    matplotlib Axes
    """
    dets = [recurrence_analysis.determinism(e, l_min) for e in epsilons]

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(epsilons, dets, '.-')
    ax.set_xlabel('epsilon')
    ax.set_ylabel('Determinism')
    ax.set_title('Determinism vs Threshold')
    return ax


# ── phase-space embedding ──────────────────────────────────────────

def plot_embedding_2d(embedded, ax=None, **kwargs):
    """Plot a 2-D delay-coordinate embedding.

    Parameters
    ----------
    embedded : ndarray, shape (M, d)
        Embedded phase-space array.  The first two columns are used.
    ax : matplotlib Axes, optional
    **kwargs
        Forwarded to ``ax.plot``.

    Returns
    -------
    matplotlib Axes
    """
    if ax is None:
        fig, ax = plt.subplots()

    kw = dict(marker='.', linestyle='none', markersize=0.5)
    kw.update(kwargs)
    ax.plot(embedded[:, 0], embedded[:, 1], **kw)
    ax.set_xlabel('x(t)')
    ax.set_ylabel('x(t + tau)')
    ax.set_title('Delay Embedding (2-D)')
    return ax


def plot_embedding_3d(embedded, ax=None, **kwargs):
    """Plot a 3-D delay-coordinate embedding.

    Parameters
    ----------
    embedded : ndarray, shape (M, d)
        Embedded array.  The first three columns are used.
    ax : mpl_toolkits.mplot3d.Axes3D, optional
    **kwargs
        Forwarded to ``ax.plot``.

    Returns
    -------
    Axes3D
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    kw = dict(linewidth=0.4)
    kw.update(kwargs)
    ax.plot(embedded[:, 0], embedded[:, 1], embedded[:, 2], **kw)
    ax.set_xlabel('x(t)')
    ax.set_ylabel('x(t + tau)')
    ax.set_zlabel('x(t + 2*tau)')
    ax.set_title('Delay Embedding (3-D)')
    return ax


def plot_autocorrelation(series, max_lag=50, optimal_delay=None, ax=None):
    """Plot the autocorrelation function with optional delay marker.

    Parameters
    ----------
    series : array-like
        1-D time series.
    max_lag : int
        Maximum lag to compute.
    optimal_delay : int, optional
        If given, a vertical line is drawn at this lag.
    ax : matplotlib Axes, optional

    Returns
    -------
    matplotlib Axes
    """
    x = np.asarray(series, dtype=float)
    mu = np.mean(x)
    var = np.sum((x - mu) ** 2)

    if ax is None:
        fig, ax = plt.subplots()

    if var == 0:
        ax.axhline(0, color='k', linewidth=0.5)
        return ax

    lags = np.arange(0, min(max_lag + 1, len(x)))
    acf = np.array([np.sum((x[:-l] - mu) * (x[l:] - mu)) / var
                     if l > 0 else 1.0
                     for l in lags])

    ax.plot(lags, acf, '.-', markersize=3)
    ax.axhline(0, color='k', linewidth=0.5, linestyle='--')

    if optimal_delay is not None:
        ax.axvline(optimal_delay, color='red', linewidth=0.8, linestyle='--',
                   label=f'tau = {optimal_delay}')
        ax.legend(fontsize=8)

    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')
    ax.set_title('Autocorrelation Function')
    return ax


# ── rolling diagnostics ────────────────────────────────────────────

def plot_diagnostics_summary(series, window_size=200, step=50, ax=None):
    """Plot Hurst exponent and Shannon entropy over rolling windows.

    Parameters
    ----------
    series : array-like
        1-D time series.
    window_size : int
        Width of the rolling window.
    step : int
        Stride between windows.
    ax : ndarray of Axes or None
        If *None*, a (2, 1) subplot figure is created.

    Returns
    -------
    ndarray of matplotlib Axes
    """
    from nonlinear.diagnostics.time_series_diagnostics import (
        TimeSeriesDiagnostics,
    )

    x = np.asarray(series, dtype=float)
    starts = list(range(0, len(x) - window_size + 1, step))
    centers = [s + window_size // 2 for s in starts]

    hursts = []
    entropies = []
    for s in starts:
        window = x[s: s + window_size]
        hursts.append(TimeSeriesDiagnostics.hurst_exponent(window))
        entropies.append(TimeSeriesDiagnostics.complexity(window))

    if ax is None:
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 5))

    ax[0].plot(centers, hursts, linewidth=0.8)
    ax[0].axhline(0.5, color='red', linewidth=0.5, linestyle='--')
    ax[0].set_ylabel('Hurst Exponent')
    ax[0].set_title('Rolling Diagnostics')

    ax[1].plot(centers, entropies, linewidth=0.8)
    ax[1].set_xlabel('Time Index')
    ax[1].set_ylabel('Shannon Entropy')

    return ax
