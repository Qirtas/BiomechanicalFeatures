import numpy as np
from scipy.signal import welch

def _rms(x):
    return float(np.sqrt(np.mean(x**2))) if x.size else np.nan

def _waveform_length(x):
    return float(np.sum(np.abs(np.diff(x))))

def _zero_crossings(x):
    return int(np.sum(np.diff(np.signbit(x)) != 0))

def _mean_freq_and_med_freq(x, fs):
    # Welch PSD; return mean freq (MNF) and median freq (MDF)
    f, Pxx = welch(x, fs=fs, nperseg=min(1024, x.size))
    if np.all(Pxx == 0):
        return np.nan, np.nan
    mnf = float(np.sum(f * Pxx) / np.sum(Pxx))
    cumsum = np.cumsum(Pxx)
    mdf = float(np.interp(0.5 * cumsum[-1], cumsum, f))
    return mnf, mdf

def _sma(acc):  # simple magnitude area of acc
    return float(np.mean(np.abs(acc), axis=1).sum())

def _jerk_mean(acc, fs):
    # jerk magnitude mean
    dif = np.diff(acc, axis=1) * fs
    mag = np.sqrt(np.sum(dif**2, axis=0))
    return float(np.mean(mag))

def compute_features_basic(window, fs_emg=None, fs_imu=None):
    feats = {}

    # --- EMG (per-channel and simple aggregates) ---
    if "emg" in window:
        emg = window["emg"]  # (ch, n)
        ch_vals_rms, ch_vals_wl, ch_vals_zc = [], [], []
        mnfs, mdfs = [], []
        for i, ch in enumerate(emg):
            ch_vals_rms.append(_rms(ch))
            ch_vals_wl.append(_waveform_length(ch))
            ch_vals_zc.append(_zero_crossings(ch))
            if fs_emg:
                mnf, mdf = _mean_freq_and_med_freq(ch, fs_emg)
                mnfs.append(mnf); mdfs.append(mdf)

            feats[f"emg_rms_ch{i}"] = ch_vals_rms[-1]
            feats[f"emg_wl_ch{i}"]  = ch_vals_wl[-1]
            feats[f"emg_zc_ch{i}"]  = ch_vals_zc[-1]

        feats["emg_rms_mean"] = float(np.mean(ch_vals_rms))
        feats["emg_wl_mean"]  = float(np.mean(ch_vals_wl))
        feats["emg_zc_mean"]  = float(np.mean(ch_vals_zc))
        if mnfs and mdfs:
            feats["emg_mnf_mean"] = float(np.mean(mnfs))
            feats["emg_mdf_mean"] = float(np.mean(mdfs))

    # --- IMU Acc ---
    if "acc" in window:
        acc = window["acc"]  # (3, n)
        feats["imu_sma"] = _sma(acc)
        if fs_imu:
            feats["imu_jerk_mean"] = _jerk_mean(acc, fs_imu)

    return feats
