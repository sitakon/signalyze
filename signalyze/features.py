"""
Feature extraction from time-domain signals and power spectral densities.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import kurtosis, skew

from .spectral import compute_welch

# Default physiological / general-purpose frequency bands.
DEFAULT_BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 100.0),
}


def extract_time_features(signal: np.ndarray) -> Dict[str, float]:
    """
    Compute time-domain statistical features from a 1-D signal.

    Parameters
    ----------
    signal : 1-D array

    Returns
    -------
    features : dict mapping feature name → float
        mean, std, variance, skewness, kurtosis, rms, peak_to_peak,
        zero_crossing_rate
    """
    x = np.asarray(signal, dtype=np.float64).ravel()
    return {
        "mean":                float(np.mean(x)),
        "std":                 float(np.std(x)),
        "variance":            float(np.var(x)),
        "skewness":            float(skew(x)),
        "kurtosis":            float(kurtosis(x)),
        "rms":                 float(np.sqrt(np.mean(x ** 2))),
        "peak_to_peak":        float(np.ptp(x)),
        "zero_crossing_rate":  float(np.sum(np.diff(np.sign(x)) != 0) / len(x)),
    }


def extract_spectral_features(
    freqs: np.ndarray,
    psd: np.ndarray,
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict[str, float]:
    """
    Compute spectral features from a PSD curve.

    Parameters
    ----------
    freqs : 1-D array
        Frequency bins in Hz (from :func:`compute_welch`).
    psd : 1-D array
        Power spectral density values.
    bands : dict, optional
        Mapping of ``band_name → (f_low, f_high)`` in Hz.
        Defaults to :data:`DEFAULT_BANDS`.

    Returns
    -------
    features : dict
        total_power, band_power_<name>, rel_power_<name> for each band,
        spectral_centroid, peak_frequency, spectral_edge_95,
        spectral_entropy, spectral_flatness.
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    psd = np.asarray(psd, dtype=np.float64)

    if bands is None:
        bands = DEFAULT_BANDS

    total_power = float(np.trapz(psd, freqs))
    features: Dict[str, float] = {"total_power": total_power}

    # Band power and relative band power
    for name, (f_lo, f_hi) in bands.items():
        mask = (freqs >= f_lo) & (freqs <= f_hi)
        bp = float(np.trapz(psd[mask], freqs[mask])) if mask.any() else 0.0
        features[f"band_power_{name}"] = bp
        features[f"rel_power_{name}"] = bp / total_power if total_power > 0 else 0.0

    psd_sum = np.sum(psd)

    # Spectral centroid
    features["spectral_centroid"] = (
        float(np.sum(freqs * psd) / psd_sum) if psd_sum > 0 else 0.0
    )

    # Peak frequency
    features["peak_frequency"] = float(freqs[np.argmax(psd)])

    # Spectral edge frequency (95 % of total power)
    cumulative = np.cumsum(psd)
    cumulative = cumulative / cumulative[-1]
    sef_idx = int(np.searchsorted(cumulative, 0.95))
    features["spectral_edge_95"] = float(freqs[min(sef_idx, len(freqs) - 1)])

    # Spectral entropy (normalised Shannon entropy of PSD treated as pmf)
    p = psd / psd_sum if psd_sum > 0 else np.ones_like(psd) / len(psd)
    p = p[p > 0]
    features["spectral_entropy"] = float(
        -np.sum(p * np.log2(p)) / np.log2(len(freqs))
    )

    # Spectral flatness (Wiener entropy): geometric mean / arithmetic mean
    log_mean = np.mean(np.log(psd + 1e-12))
    features["spectral_flatness"] = float(
        np.exp(log_mean) / (np.mean(psd) + 1e-12)
    )

    return features


def extract_all_features(
    signal: np.ndarray,
    fs: float,
    nperseg: Optional[int] = None,
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
    include_time: bool = True,
    include_spectral: bool = True,
) -> Dict[str, float]:
    """
    Extract time and/or spectral features from a single signal.

    Parameters
    ----------
    signal : 1-D array
    fs : float
        Sampling frequency in Hz.
    nperseg : int, optional
        Welch segment length.
    bands : dict, optional
        Custom frequency bands for spectral features.
    include_time : bool, default True
    include_spectral : bool, default True

    Returns
    -------
    features : dict
    """
    features: Dict[str, float] = {}

    if include_time:
        features.update(extract_time_features(signal))

    if include_spectral:
        freqs, psd = compute_welch(signal, fs, nperseg=nperseg)
        features.update(extract_spectral_features(freqs, psd, bands=bands))

    return features


def features_to_matrix(
    feature_dicts: List[Dict[str, float]],
) -> Tuple[np.ndarray, List[str]]:
    """
    Stack a list of feature dicts into a 2-D feature matrix.

    All dicts must have the same keys (and key order is taken from the first).

    Parameters
    ----------
    feature_dicts : list of dict

    Returns
    -------
    X : ndarray, shape (n_samples, n_features)
    feature_names : list of str
    """
    if not feature_dicts:
        raise ValueError("feature_dicts is empty")
    feature_names = list(feature_dicts[0].keys())
    X = np.array([[d[k] for k in feature_names] for d in feature_dicts],
                 dtype=np.float64)
    return X, feature_names
