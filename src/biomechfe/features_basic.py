# src/biomechfe/features_basic.py
import numpy as np
from scipy.signal import welch
from .Featureset.IMU_acc_3axis_features import compute_acc_features_for_window
from .Featureset.IMU_acc_jerk_features import compute_jerk_features_for_window
from .Featureset.IMU_gyr_3axis_features import compute_gyr_features_for_window
from .Featureset.IMU_gyr_rom_features import compute_gyr_rom_features_for_window
from .Featureset.IMU_movement_frequency_features import compute_movement_frequency_features_for_window
from .Featureset.IMU_MPSD_features import compute_mpsd_features_for_window
from .Featureset.IMU_RT_variability_features import compute_rt_variability_features_for_window
from .Featureset.EMG_integrated_features import compute_emg_integrated_features_for_window
from .Featureset.EMG_MAV_features import compute_emg_mav_features_for_window
from .Featureset.EMG_statistical_features import compute_emg_statistical_features_for_window
from .Featureset.EMG_STFT_features import compute_emg_stft_features_for_window
from .Featureset.EMG_wavelength_features import compute_emg_wavelength_features_for_window
from .Featureset.EMG_ZC_features import compute_emg_zc_features_for_window

def compute_features_basic(window, fs_emg=None, fs_imu=None, imu_site=None, muscle_names=None, extended_acc=False, include_jerk=True):
    """
    Compute basic features from a window of EMG and/or IMU data.

    Parameters:
    -----------
    window : dict
        Window containing 'emg' and/or 'acc'/'gyr' data
    fs_emg : float, optional
        EMG sampling frequency
    fs_imu : float, optional
        IMU sampling frequency
    imu_site : str, optional
        IMU sensor site name (e.g., 'Shoulder', 'Wrist') for feature naming
    extended_acc : bool, default False
        Whether to use extended accelerometer features (more comprehensive)
    include_jerk : bool, default True
        Whether to include jerk features from accelerometer data

    Returns:
    --------
    dict
        Dictionary of computed features
    """
    feats = {}

    # --- EMG (per-channel and simple aggregates) ---
    if "emg" in window and window["emg"] is not None:
        emg_integrated_features = compute_emg_integrated_features_for_window(
            window, fs_emg, muscle_names=muscle_names
        )
        feats.update(emg_integrated_features)

        emg_MAV_features = compute_emg_mav_features_for_window(
            window, fs_emg, muscle_names=muscle_names
        )
        feats.update(emg_MAV_features)

        emg_statistical_features = compute_emg_statistical_features_for_window(
            window, fs_emg, muscle_names=muscle_names
        )
        feats.update(emg_statistical_features)

        emg_STFT_features = compute_emg_stft_features_for_window(
            window, fs_emg, muscle_names=muscle_names
        )
        feats.update(emg_STFT_features)

        emg_wavelength_features = compute_emg_wavelength_features_for_window(
            window, fs_emg, muscle_names=muscle_names
        )
        feats.update(emg_wavelength_features)

        emg_zc_features = compute_emg_zc_features_for_window(
            window, fs_emg, muscle_names=muscle_names
        )
        feats.update(emg_zc_features)

    # --- IMU Accelerometer (comprehensive 3-axis features) ---
    if "acc" in window and window["acc"] is not None and fs_imu:
        # Use the comprehensive 3-axis accelerometer features
        acc_features = compute_acc_features_for_window(
            window, fs_imu, site_name=imu_site, extended=extended_acc
        )
        feats.update(acc_features)

        jerk_features = compute_jerk_features_for_window(
            window, fs_imu, site_name=imu_site
        )
        feats.update(jerk_features)

        movement_freq_acc_features = compute_movement_frequency_features_for_window(
            window, fs_imu, site_name=imu_site
        )
        feats.update(movement_freq_acc_features)

        mpsd_acc_features = compute_mpsd_features_for_window(
            window, fs_imu, site_name=imu_site
        )
        feats.update(mpsd_acc_features)

        acc_rt_variability_features = compute_rt_variability_features_for_window(
            window, fs_imu, site_name=imu_site
        )
        feats.update(acc_rt_variability_features)

    # --- IMU Gyroscope (basic features - you can extend this similarly) ---
    if "gyr" in window and window["gyr"] is not None:
        gyr = window["gyr"]  # (3, n)
        # Add basic gyroscope features here
        # You could create a similar comprehensive gyroscope feature module
        gyr_3axisfeatures = compute_gyr_features_for_window(
            window, fs_imu, site_name=imu_site
        )
        feats.update(gyr_3axisfeatures)

        gyr_romfeatures = compute_gyr_rom_features_for_window(
            window, fs_imu, site_name=imu_site
        )
        feats.update(gyr_romfeatures)

        movement_freq_gyr_features = compute_movement_frequency_features_for_window(
            window, fs_imu, site_name=imu_site
        )
        feats.update(movement_freq_gyr_features)

        mpsd_gyr_features = compute_mpsd_features_for_window(
            window, fs_imu, site_name=imu_site
        )
        feats.update(mpsd_gyr_features)

        gyr_rt_variability_features = compute_rt_variability_features_for_window(
            window, fs_imu, site_name=imu_site
        )
        feats.update(gyr_rt_variability_features)

    return feats