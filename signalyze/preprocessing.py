"""
Signal preprocessing: segmentation and filtering.
"""

from typing import List, Literal, Optional, Tuple, Union

import numpy as np
from scipy.signal import butter, cheby1, cheby2, sosfiltfilt

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

FilterType = Literal["butter", "cheby1", "cheby2"]
BType = Literal["lowpass", "highpass", "bandpass"]
Cutoff = Union[float, Tuple[float, float]]


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------


def segment_signal(
    signal: np.ndarray,
    segment_length: int,
    overlap: float = 0.0,
    drop_last: bool = True,
) -> List[np.ndarray]:
    """
    Split a 1-D signal into fixed-length segments.

    Parameters
    ----------
    signal : 1-D array
        Input time-series.
    segment_length : int
        Number of samples per segment.
    overlap : float, default 0.0
        Fractional overlap between consecutive segments, in [0, 1).
        E.g. 0.5 means each segment shares half its samples with the next.
    drop_last : bool, default True
        If True, discard the final segment when it would be shorter than
        ``segment_length``. If False, keep it (zero-padded).

    Returns
    -------
    segments : list of ndarray, each of shape (segment_length,)
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()
    if segment_length < 1:
        raise ValueError("segment_length must be >= 1")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")

    step = max(1, int(segment_length * (1.0 - overlap)))
    starts = range(0, len(signal), step)
    segments = []

    for start in starts:
        end = start + segment_length
        chunk = signal[start:end]
        if len(chunk) < segment_length:
            if drop_last:
                break
            chunk = np.pad(chunk, (0, segment_length - len(chunk)))
        segments.append(chunk)

    return segments


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def _build_sos(
    btype: BType,
    cutoff: Cutoff,
    fs: float,
    order: int,
    filter_type: FilterType,
    rp: float,
    rs: float,
) -> np.ndarray:
    """Return a second-order-sections array for the requested filter."""
    nyq = fs / 2.0

    if btype == "bandpass":
        if not (isinstance(cutoff, (list, tuple)) and len(cutoff) == 2):
            raise ValueError("bandpass requires cutoff=(low_hz, high_hz)")
        wn = (cutoff[0] / nyq, cutoff[1] / nyq)
    else:
        if not isinstance(cutoff, (int, float)):
            raise ValueError(f"{btype} requires a scalar cutoff frequency")
        wn = cutoff / nyq

    if filter_type == "butter":
        return butter(order, wn, btype=btype, output="sos")
    elif filter_type == "cheby1":
        return cheby1(order, rp, wn, btype=btype, output="sos")
    elif filter_type == "cheby2":
        return cheby2(order, rs, wn, btype=btype, output="sos")
    else:
        raise ValueError(f"Unknown filter_type '{filter_type}'. Use 'butter', 'cheby1', or 'cheby2'.")


def filter_signal(
    signal: np.ndarray,
    fs: float,
    btype: BType,
    cutoff: Cutoff,
    order: int = 4,
    filter_type: FilterType = "butter",
    rp: float = 1.0,
    rs: float = 40.0,
) -> np.ndarray:
    """
    Apply a zero-phase digital filter to a 1-D signal.

    Uses second-order sections (SOS) representation for numerical stability
    and ``sosfiltfilt`` for zero-phase (forward-backward) filtering.

    Parameters
    ----------
    signal : 1-D array
        Input time-series.
    fs : float
        Sampling frequency in Hz.
    btype : {'lowpass', 'highpass', 'bandpass'}
        Filter pass-band type.
    cutoff : float or (float, float)
        Cutoff frequency in Hz. Provide a scalar for lowpass/highpass,
        or a ``(low, high)`` tuple for bandpass.
    order : int, default 4
        Filter order. Higher order → steeper roll-off but more ringing.
    filter_type : {'butter', 'cheby1', 'cheby2'}, default 'butter'
        Filter family:

        - ``'butter'``  — Butterworth (maximally flat passband).
        - ``'cheby1'``  — Chebyshev Type I (equiripple passband, monotone
          stopband). Controlled by ``rp``.
        - ``'cheby2'``  — Chebyshev Type II (monotone passband, equiripple
          stopband). Controlled by ``rs``.

    rp : float, default 1.0
        Maximum passband ripple in dB. Used only by ``'cheby1'``.
    rs : float, default 40.0
        Minimum stopband attenuation in dB. Used only by ``'cheby2'``.

    Returns
    -------
    filtered : ndarray
        Filtered signal with the same length as ``signal``.
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()
    sos = _build_sos(btype, cutoff, fs, order, filter_type, rp, rs)
    return sosfiltfilt(sos, signal)


def preprocess_signal(
    signal: np.ndarray,
    fs: float,
    btype: Optional[BType] = None,
    cutoff: Optional[Cutoff] = None,
    order: int = 4,
    filter_type: FilterType = "butter",
    rp: float = 1.0,
    rs: float = 40.0,
    segment_length: Optional[int] = None,
    overlap: float = 0.0,
    drop_last: bool = True,
) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Convenience wrapper: optionally filter then optionally segment a signal.

    Parameters
    ----------
    signal : 1-D array
        Raw input time-series.
    fs : float
        Sampling frequency in Hz.
    btype : {'lowpass', 'highpass', 'bandpass'}, optional
        If provided, the signal is filtered before segmentation.
    cutoff : float or (float, float), optional
        Cutoff frequency / frequencies in Hz. Required when ``btype`` is set.
    order : int, default 4
        Filter order.
    filter_type : {'butter', 'cheby1', 'cheby2'}, default 'butter'
    rp : float, default 1.0
        Chebyshev Type I passband ripple (dB).
    rs : float, default 40.0
        Chebyshev Type II stopband attenuation (dB).
    segment_length : int, optional
        If provided, the (possibly filtered) signal is split into segments of
        this many samples.
    overlap : float, default 0.0
        Fractional overlap between segments, in [0, 1).
    drop_last : bool, default True
        Discard the final short segment if True, zero-pad it if False.

    Returns
    -------
    ndarray
        Filtered signal (when ``segment_length`` is None).
    list of ndarray
        Segments (when ``segment_length`` is given).
    """
    sig = np.asarray(signal, dtype=np.float64).ravel()

    if btype is not None:
        if cutoff is None:
            raise ValueError("cutoff must be provided when btype is set")
        sig = filter_signal(sig, fs, btype=btype, cutoff=cutoff, order=order,
                            filter_type=filter_type, rp=rp, rs=rs)

    if segment_length is not None:
        return segment_signal(sig, segment_length, overlap=overlap, drop_last=drop_last)

    return sig
