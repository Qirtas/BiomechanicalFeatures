# src/biomechfe/segmentation.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Literal, Optional
import numpy as np
from scipy.signal import find_peaks, savgol_filter

@dataclass(frozen=True)
class RepParams:
    """
    Parameters for repetition segmentation from a gyroscope signal.
    """
    site: Optional[str] = None           # just metadata; selection happens in your loader
    axis: Optional[int] = None           # 0/1/2 → X/Y/Z; None = auto-pick by variance
    smooth_s: float = 0.20               # Savitzky–Golay smoothing window (seconds)
    smooth_poly: int = 3
    min_prominence: float = 0.20         # relative to signal std after smoothing
    min_distance_s: float = 0.80         # seconds between peaks/valleys (≈ cadence)
    mode: Literal[
        "zero_between_valley_peak",      # internal rotation style
        "zero_between_peak_valley",      # external rotation style
        "peak_valley_pairs"              # flexion/extension style
    ] = "peak_valley_pairs"

def _choose_axis_by_variance(gyr: np.ndarray) -> int:
    # gyr: (3, n)
    var = gyr.var(axis=1)
    return int(np.argmax(var))

def _smooth(sig: np.ndarray, fs: float, win_s: float, poly: int) -> np.ndarray:
    n = sig.size
    win = max(5, int(round(win_s * fs)) | 1)  # odd ≥5
    win = min(win, n - 1 if (n % 2 == 0) else n)  # ensure <= n and odd
    if win < 5:  # too short to smooth
        return sig
    return savgol_filter(sig, window_length=win, polyorder=min(poly, win - 1))

def _zero_crossings(sig: np.ndarray) -> np.ndarray:
    signs = np.signbit(sig)
    idx = np.where(np.diff(signs) != 0)[0]
    # interpolate to sub-sample? Keep simple: next index after sign change
    return idx + 1

def _find_peaks_and_valleys(sig: np.ndarray, fs: float, prom: float, dist_s: float):
    distance = max(1, int(dist_s * fs))
    # normalize prominence by std
    scale = np.std(sig) if np.std(sig) > 0 else 1.0
    peaks, _ = find_peaks(sig, prominence=prom * scale, distance=distance)
    valleys, _ = find_peaks(-sig, prominence=prom * scale, distance=distance)
    return peaks, valleys

def segment_repetitions_from_gyr(
    gyr: np.ndarray, fs: float, params: RepParams
) -> List[Tuple[int, int]]:
    """
    Returns list of (start_idx, end_idx) in IMU sample indices for each repetition.
    gyr: (3, n) gyroscope array after your 20 Hz low-pass preprocessing.
    """
    if gyr is None or gyr.ndim != 2 or gyr.shape[0] != 3:
        return []

    # pick axis
    ax = params.axis if params.axis is not None else _choose_axis_by_variance(gyr)
    sig = gyr[ax].astype(float)

    # smooth
    s = _smooth(sig, fs, params.smooth_s, params.smooth_poly)

    # peaks, valleys, zero crossings
    peaks, valleys = _find_peaks_and_valleys(s, fs, params.min_prominence, params.min_distance_s)
    zeros = _zero_crossings(s)

    spans: List[Tuple[int, int]] = []

    if params.mode.startswith("zero_between"):
        # Build ordered pairs valley→peak or peak→valley
        if params.mode == "zero_between_valley_peak":
            a, b = valleys, peaks
        else:
            a, b = peaks, valleys
        # For each a→b, find zero-crossing pairs inside and emit spans between successive zeros
        j = 0
        for i in range(len(a)):
            # next b after a[i]
            b_after = b[b > a[i]]
            if b_after.size == 0:
                break
            right = b_after[0]
            # zeros between a[i] and right
            inband = zeros[(zeros > a[i]) & (zeros < right)]
            # consecutive zero pairs
            for k in range(len(inband) - 1):
                start, end = int(inband[k]), int(inband[k + 1])
                if end > start:
                    spans.append((start, end))

    else:  # "peak_valley_pairs"
        # pair peak→next valley (or valley→next peak if that fits better for your dataset)
        # here we do symmetric: create both lists and choose the pairing that yields more spans
        def pair_forward(first, second):
            spans_local = []
            for p in first:
                q = second[second > p]
                if q.size:
                    spans_local.append((int(p), int(q[0])))
            return spans_local

        pv = pair_forward(peaks, valleys)
        vp = pair_forward(valleys, peaks)
        spans = pv if len(pv) >= len(vp) else vp

    # filter out too-short spans (< 0.3s)
    min_len = int(0.30 * fs)
    spans = [(a, b) for a, b in spans if (b - a) >= min_len]

    return spans
