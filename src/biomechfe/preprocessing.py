# src/biomechfe/preprocessing.py
from __future__ import annotations
import numpy as np
from scipy.signal import butter, sosfiltfilt, iirnotch

# ---------------------------
# Utilities
# ---------------------------

def _as_2d(x: np.ndarray) -> np.ndarray:
    """Ensure array is (channels, samples)."""
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[np.newaxis, :]
    if x.ndim != 2:
        raise ValueError("Input must be 1D or 2D array shaped (channels, samples).")
    return x

def _can_filt(x_len: int, filt_len: int) -> bool:
    # sosfiltfilt padlen ~ 3 * (max(len(a), len(b)) - 1); with SOS it’s 3 * 2 = 6 by default,
    # but we add a conservative guard. If too short, we’ll fall back to no filtering.
    return x_len > max(21, 3 * filt_len)

# ---------------------------
# EMG preprocessing
# ---------------------------

def preprocess_emg(
    emg: np.ndarray | None,
    fs: float | None,
    band: tuple[float, float] = (30.0, 350.0),
    order: int = 4,
    notch_hz: list[float] | None = None,   # e.g., [50.0] or [60.0] if needed
) -> dict | None:
    """
    Mean-remove then Butterworth band-pass (default 30–350 Hz).
    Optionally apply mains notch(es) before band-pass.

    Returns a dict: {"emg": np.ndarray (channels, samples)} to match the package API.
    """
    if emg is None:
        return None
    if fs is None:
        raise ValueError("fs is required for EMG preprocessing.")

    x = _as_2d(np.asarray(emg, dtype=float))

    # 1) mean removal (per channel)
    x = x - x.mean(axis=1, keepdims=True)

    # 2) optional notch(es)
    if notch_hz:
        for f0 in notch_hz:
            # Quality factor: narrow notch (adjust if needed)
            Q = 30.0
            b_notch, a_notch = iirnotch(w0=f0/(fs/2), Q=Q)
            # sosfiltfilt expects SOS; convert quickly via butter? Instead, use filtfilt via sosfiltfilt-like pad.
            # Simpler: apply filtfilt with (b, a) through numpy per channel for robustness.
            # But to keep dependencies consistent, we can emulate per channel:
            from scipy.signal import filtfilt
            # Apply per channel
            x = np.vstack([filtfilt(b_notch, a_notch, ch, method="pad") for ch in x])

    # 3) band-pass via SOS for numerical stability
    low, high = band
    if not (0 < low < high < fs/2):
        raise ValueError(f"Invalid EMG band {band} for fs={fs}.")
    sos = butter(order, [low/(fs/2), high/(fs/2)], btype="bandpass", output="sos")

    # Guard extremely short clips
    n = x.shape[1]
    if _can_filt(n, filt_len=order*2):
        x = np.vstack([sosfiltfilt(sos, ch) for ch in x])
    # else: leave as mean-removed (still valid)

    return {"emg": x}

# ---------------------------
# IMU preprocessing
# ---------------------------

def preprocess_imu(
    imu: dict | None,
    fs: float | None,
    cutoff_hz: float = 20.0,
    order: int = 4,
) -> dict | None:
    """
    Mean-remove per axis (Acc_X/Y/Z, Gyr_X/Y/Z), then Butterworth low-pass (default 20 Hz).
    Expects imu dict like {"acc": (3, n), "gyr": (3, n)}; keys missing are fine.
    """
    if imu is None:
        return None
    if fs is None:
        raise ValueError("fs is required for IMU preprocessing.")

    out = {}
    sos = None
    if not (0 < cutoff_hz < fs/2):
        raise ValueError(f"Invalid IMU cutoff {cutoff_hz} for fs={fs}.")
    sos = butter(order, cutoff_hz/(fs/2), btype="lowpass", output="sos")

    for key in ("acc", "gyr", "mag"):
        arr = imu.get(key)
        if arr is None:
            out[key] = None
            continue
        x = np.asarray(arr, dtype=float)
        # Expect (3, n). Accept (n, 3) and transpose if needed.
        if x.ndim == 2 and x.shape[0] != 3 and x.shape[1] == 3:
            x = x.T
        if x.ndim != 2 or x.shape[0] != 3:
            raise ValueError(f"IMU {key} must be shaped (3, samples) or (samples, 3).")
        # mean removal per axis
        x = x - x.mean(axis=1, keepdims=True)
        # filter (guard short clips)
        n = x.shape[1]
        if _can_filt(n, filt_len=order*2):
            x = np.vstack([sosfiltfilt(sos, axis) for axis in x])
        out[key] = x

    return out
