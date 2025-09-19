import numpy as np
from scipy.signal import butter, filtfilt

def _butter_bandpass(low, high, fs, order=4):
    return butter(order, [low/(fs/2), high/(fs/2)], btype="band")

def preprocess_emg(emg, fs):
    if emg is None: return None
    # very basic: 20–450 Hz band-pass + rectify + 6 Hz low-pass envelope
    b, a = _butter_bandpass(20, 450, fs)
    bp = filtfilt(b, a, emg, axis=1)
    rect = np.abs(bp)
    # envelope (simple moving average to keep deps light)
    win = max(1, int(fs/6))
    kernel = np.ones(win)/win
    env = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode="same"), 1, rect)
    return {"emg": env}

def preprocess_imu(imu, fs):
    if imu is None: return None
    # assume imu = {"acc": (3,n), "gyr": (3,n)}; simple detrend
    out = {}
    for k, v in imu.items():
        if v is None:
            out[k] = None
        else:
            v = v - v.mean(axis=1, keepdims=True)
            out[k] = v
    return out
