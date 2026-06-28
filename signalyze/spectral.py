"""
Welch periodogram computation and visualization.
"""

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch


def compute_welch(
    signal: np.ndarray,
    fs: float,
    nperseg: Optional[int] = None,
    noverlap: Optional[int] = None,
    window: str = "hann",
    scaling: str = "density",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the Welch power spectral density estimate.

    Parameters
    ----------
    signal : 1-D array
        Input time-series signal.
    fs : float
        Sampling frequency in Hz.
    nperseg : int, optional
        Length of each segment. Defaults to min(len(signal), 256).
    noverlap : int, optional
        Number of overlapping samples between segments. Defaults to nperseg // 2.
    window : str, default 'hann'
        Window function applied to each segment.
    scaling : {'density', 'spectrum'}, default 'density'
        'density' returns V²/Hz; 'spectrum' returns V².

    Returns
    -------
    freqs : ndarray
        Array of sample frequencies.
    psd : ndarray
        Power spectral density or power spectrum.
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()
    if nperseg is None:
        nperseg = min(len(signal), 256)
    freqs, psd = welch(
        signal,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        window=window,
        scaling=scaling,
    )
    return freqs, psd


def plot_welch(
    signal: np.ndarray,
    fs: float,
    ax: Optional[plt.Axes] = None,
    nperseg: Optional[int] = None,
    log_scale: bool = True,
    label: Optional[str] = None,
    title: Optional[str] = None,
    color: Optional[str] = None,
) -> plt.Axes:
    """
    Plot the Welch PSD of a single signal.

    Parameters
    ----------
    signal : 1-D array
        Input time-series signal.
    fs : float
        Sampling frequency in Hz.
    ax : matplotlib Axes, optional
        Axes to draw on. A new figure is created if None.
    nperseg : int, optional
        Segment length for Welch estimate.
    log_scale : bool, default True
        Use log scale on the power axis.
    label : str, optional
        Legend label for the line.
    title : str, optional
        Plot title.
    color : str, optional
        Line color.

    Returns
    -------
    ax : matplotlib Axes
    """
    freqs, psd = compute_welch(signal, fs, nperseg=nperseg)

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    kwargs = {}
    if label is not None:
        kwargs["label"] = label
    if color is not None:
        kwargs["color"] = color

    (ax.semilogy if log_scale else ax.plot)(freqs, psd, **kwargs)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (V²/Hz)" if log_scale else "PSD")
    ax.set_title(title or "Welch Periodogram")
    ax.grid(True, alpha=0.3)
    if label is not None:
        ax.legend()

    return ax


def compute_percentile_psd(
    signals: List[np.ndarray],
    fs: float,
    percentiles: Tuple[float, ...] = (5, 25, 50, 75, 95),
    nperseg: Optional[int] = None,
    noverlap: Optional[int] = None,
    window: str = "hann",
    scaling: str = "density",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-frequency percentiles of the PSD across many signals.

    Parameters
    ----------
    signals : list of 1-D arrays
        Collection of time-series signals (must share the same sampling rate).
    fs : float
        Sampling frequency in Hz.
    percentiles : tuple of float, default (5, 25, 50, 75, 95)
        Percentile values to compute at each frequency bin.
    nperseg : int, optional
        Segment length for Welch estimate.
    noverlap : int, optional
        Overlap between segments.
    window : str, default 'hann'
    scaling : str, default 'density'

    Returns
    -------
    freqs : ndarray, shape (F,)
        Frequency bins.
    psd_percentiles : ndarray, shape (len(percentiles), F)
        Rows correspond to ``percentiles`` in the same order.
    """
    psds = []
    for sig in signals:
        freqs, psd = compute_welch(
            sig, fs, nperseg=nperseg, noverlap=noverlap, window=window, scaling=scaling
        )
        psds.append(psd)
    psd_matrix = np.stack(psds, axis=0)  # (N, F)
    psd_percentiles = np.percentile(psd_matrix, list(percentiles), axis=0)  # (P, F)
    return freqs, psd_percentiles


def plot_percentile_psd(
    signals: List[np.ndarray],
    fs: float,
    percentiles: Tuple[float, ...] = (5, 25, 50, 75, 95),
    nperseg: Optional[int] = None,
    log_scale: bool = True,
    fill_between: bool = True,
    color: str = "steelblue",
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (10, 5),
    title: Optional[str] = None,
    overlay_signal: Optional[np.ndarray] = None,
    overlay_label: Optional[str] = None,
    overlay_color: str = "crimson",
) -> plt.Axes:
    """
    Plot per-frequency percentile bands of the PSD across many signals,
    with an optional single-signal overlay.

    Parameters
    ----------
    signals : list of 1-D arrays
        Population of signals whose PSD distribution is visualised.
    fs : float
        Sampling frequency in Hz.
    percentiles : tuple of float, default (5, 25, 50, 75, 95)
        Percentiles to show. The median (middle value) is drawn as a solid
        line; adjacent pairs are filled as shaded bands. Works best with an
        odd number of values centred on 50.
    nperseg : int, optional
        Segment length for Welch estimate.
    log_scale : bool, default True
        Log-scale the power axis.
    fill_between : bool, default True
        Shade the inter-percentile bands.
    color : str, default 'steelblue'
        Base colour for the percentile bands and median line.
    ax : matplotlib Axes, optional
        Axes to draw on. A new figure is created if None.
    figsize : tuple, default (10, 5)
    title : str, optional
    overlay_signal : 1-D array, optional
        A single signal whose PSD is plotted on top of the percentile bands.
    overlay_label : str, optional
        Legend label for the overlay signal.
    overlay_color : str, default 'crimson'
        Colour for the overlay signal line.

    Returns
    -------
    ax : matplotlib Axes
    """
    freqs, psd_pct = compute_percentile_psd(
        signals, fs, percentiles=percentiles, nperseg=nperseg
    )

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    plot_fn = ax.semilogy if log_scale else ax.plot
    n_pct = len(percentiles)
    mid = n_pct // 2

    # Shaded bands: pair outer percentiles inward toward the median
    if fill_between:
        alphas = np.linspace(0.15, 0.30, mid)
        for i in range(mid):
            ax.fill_between(
                freqs,
                psd_pct[i],
                psd_pct[n_pct - 1 - i],
                alpha=alphas[i],
                color=color,
                label=f"p{percentiles[i]:.0f}–p{percentiles[n_pct-1-i]:.0f}",
            )

    # Median (or central percentile)
    plot_fn(
        freqs,
        psd_pct[mid],
        color=color,
        linewidth=1.8,
        label=f"p{percentiles[mid]:.0f} (median)",
    )

    # Optional single-signal overlay
    if overlay_signal is not None:
        _, psd_ov = compute_welch(overlay_signal, fs, nperseg=nperseg)
        plot_fn(
            freqs,
            psd_ov,
            color=overlay_color,
            linewidth=1.5,
            linestyle="--",
            label=overlay_label or "signal",
        )

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (V²/Hz)")
    ax.set_title(title or "Percentile Power Spectrum")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return ax


def plot_welch_comparison(
    signals: List[np.ndarray],
    labels: List[str],
    fs: float,
    nperseg: Optional[int] = None,
    log_scale: bool = True,
    figsize: Tuple[int, int] = (10, 5),
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Overlay Welch PSDs for multiple signals on one axes.

    Parameters
    ----------
    signals : list of 1-D arrays
        Signals to compare.
    labels : list of str
        Legend labels, one per signal.
    fs : float
        Sampling frequency in Hz.
    nperseg : int, optional
        Segment length for Welch estimate.
    log_scale : bool, default True
        Use log scale on the power axis.
    figsize : tuple, default (10, 5)
    title : str, optional

    Returns
    -------
    fig : matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.tab10.colors

    for i, (sig, lbl) in enumerate(zip(signals, labels)):
        freqs, psd = compute_welch(sig, fs, nperseg=nperseg)
        (ax.semilogy if log_scale else ax.plot)(
            freqs, psd, label=lbl, color=colors[i % len(colors)], alpha=0.85
        )

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (V²/Hz)")
    ax.set_title(title or "Welch Periodogram Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig
