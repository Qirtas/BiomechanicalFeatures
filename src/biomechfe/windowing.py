import numpy as np

def _sliding_indices(n, win, step):
    starts = range(0, max(0, n - win + 1), step)
    for s in starts:
        yield s, s + win

def make_windows(emg, imu, fs_emg, fs_imu, window_s=2.0, step_s=0.5):
    win_emg = int(window_s * fs_emg) if (emg and "emg" in emg) else None
    step_emg = int(step_s * fs_emg) if (emg and "emg" in emg) else None
    acc = imu.get("acc") if imu else None
    win_imu = int(window_s * fs_imu) if acc is not None else None
    step_imu = int(step_s * fs_imu) if acc is not None else None

    n_windows = 0
    if win_emg is not None:
        n_windows = max(n_windows, sum(1 for _ in _sliding_indices(emg["emg"].shape[1], win_emg, step_emg)))
    if win_imu is not None:
        n_windows = max(n_windows, sum(1 for _ in _sliding_indices(acc.shape[1], win_imu, step_imu)))

    windows = []
    for i in range(n_windows):
        w = {}
        if win_emg is not None:
            s = i * step_emg
            w["emg"] = emg["emg"][:, s:s+win_emg]
        if win_imu is not None:
            s = i * step_imu
            w["acc"] = acc[:, s:s+win_imu]
        windows.append(w)
    return windows
