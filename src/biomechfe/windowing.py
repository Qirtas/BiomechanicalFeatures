# src/biomechfe/windowing.py
from __future__ import annotations
import numpy as np

def _sliding_indices(n, win, step):
    starts = range(0, max(0, n - win + 1), step)
    for s in starts:
        yield s, s + win

def make_windows(
    emg, imu, fs_emg, fs_imu, window_s=2.0, step_s=0.5, allow_partial: bool = False
):
    # lengths
    emg_arr = emg["emg"] if (emg and "emg" in emg) else None
    acc = imu.get("acc") if imu else None

    win_emg = int(window_s * fs_emg) if (emg_arr is not None and fs_emg) else None
    step_emg = int(step_s * fs_emg) if (emg_arr is not None and fs_emg) else None
    win_imu = int(window_s * fs_imu) if (acc is not None and fs_imu) else None
    step_imu = int(step_s * fs_imu) if (acc is not None and fs_imu) else None

    # number of windows = max across modalities
    n_windows = 0
    if win_emg is not None:
        n_windows = max(n_windows, sum(1 for _ in _sliding_indices(emg_arr.shape[1], win_emg, step_emg)))
    if win_imu is not None:
        n_windows = max(n_windows, sum(1 for _ in _sliding_indices(acc.shape[1], win_imu, step_imu)))

    windows = []

    # allow a single short window if requested
    if n_windows == 0 and allow_partial:
        w = {}
        if emg_arr is not None:
            n = emg_arr.shape[1]
            take = n if (win_emg is None) else min(n, win_emg)
            w["emg"] = emg_arr[:, :take]
        if acc is not None:
            n = acc.shape[1]
            take = n if (win_imu is None) else min(n, win_imu)
            w["acc"] = acc[:, :take]
        if w:
            windows.append(w)
        return windows

    for i in range(n_windows):
        w = {}
        if win_emg is not None:
            s = i * step_emg
            w["emg"] = emg_arr[:, s:s + win_emg]
        if win_imu is not None:
            s = i * step_imu
            w["acc"] = acc[:, s:s + win_imu]
        windows.append(w)

    return windows


def make_windows_from_segments(
    emg, imu, fs_emg, fs_imu, segments_imu: list[tuple[int, int]]
):
    """
    Create windows that follow custom (t0,t1) segments defined in IMU sample indices.
    EMG windows are time-aligned using sampling rates.
    """
    windows = []
    for s_i, e_i in segments_imu:
        w = {}
        if imu:
            if imu.get("acc") is not None:
                w["acc"] = imu["acc"][:, s_i:e_i]
            if imu.get("gyr") is not None:
                w["gyr"] = imu["gyr"][:, s_i:e_i]
        if emg and "emg" in emg and fs_emg and fs_imu:
            t0, t1 = s_i / fs_imu, e_i / fs_imu
            s_e, e_e = int(round(t0 * fs_emg)), int(round(t1 * fs_emg))
            w["emg"] = emg["emg"][:, s_e:e_e]
        windows.append(w)
    return windows
