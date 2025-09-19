import pandas as pd
import numpy as np
from .preprocessing import preprocess_emg, preprocess_imu
from .windowing import make_windows
from .features_basic import compute_features_basic

def _validate_inputs(data, fs_emg, fs_imu):
    if ("emg" not in data) and ("imu" not in data):
        raise ValueError("data must contain 'emg' and/or 'imu'.")

    if data.get("emg") is not None:
        emg = np.asarray(data["emg"])
        if emg.ndim != 2:
            raise ValueError("EMG must be 2D array shaped (channels, samples).")
        if fs_emg is None:
            raise ValueError("fs_emg is required when EMG is provided.")

    imu = data.get("imu")
    if imu:
        acc = imu.get("acc")
        if acc is not None:
            acc = np.asarray(acc)
            if acc.shape[0] != 3:
                raise ValueError("IMU acc must be shaped (3, samples).")
            if fs_imu is None:
                raise ValueError("fs_imu is required when IMU is provided.")

def extract_features(data, fs_emg=None, fs_imu=None, window_s=2.0, step_s=0.5):
    """
    data: {"emg": np.ndarray|None, "imu": {"acc": np.ndarray, "gyr": np.ndarray}|None}
    Returns: pandas.DataFrame (rows = windows, cols = features)
    """
    _validate_inputs(data, fs_emg, fs_imu)
    emg = preprocess_emg(data.get("emg"), fs_emg) if data.get("emg") is not None else None
    imu = preprocess_imu(data.get("imu"), fs_imu) if data.get("imu") is not None else None
    windows = make_windows(emg=emg, imu=imu, fs_emg=fs_emg, fs_imu=fs_imu,
                           window_s=window_s, step_s=step_s)
    rows = [compute_features_basic(w, fs_emg=fs_emg, fs_imu=fs_imu) for w in windows]
    return pd.DataFrame(rows)
