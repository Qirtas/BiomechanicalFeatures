"""
3-axis accelerometer feature extraction for biomechanical analysis.
Computes statistical and signal features for X, Y, Z axes and magnitude.
"""

import numpy as np
from scipy.stats import skew, kurtosis
from typing import Dict, Optional


def compute_3axis_acc_features(
        acc_data: np.ndarray,
        fs: float,
        site_name: Optional[str] = None
) -> Dict[str, float]:
    """
    Compute comprehensive 3-axis accelerometer features.

    Parameters:
    -----------
    acc_data : np.ndarray
        Accelerometer data shaped (3, n_samples) where rows are [X, Y, Z]
    fs : float
        Sampling frequency in Hz
    site_name : str, optional
        Name of the sensor site (e.g., 'Shoulder', 'Wrist') for feature naming.
        If None, uses generic naming.

    Returns:
    --------
    Dict[str, float]
        Dictionary containing features with descriptive names:
        - Individual axis features: X_{site}_acc_Mean, Y_{site}_acc_Std, etc.
        - Magnitude features: Magnitude_{site}_acc_Mean, etc.
        If site_name is None, uses format: X_acc_Mean, Magnitude_acc_Mean, etc.
    """
    if acc_data.shape[0] != 3:
        raise ValueError(f"Expected acc_data shape (3, n_samples), got {acc_data.shape}")

    features = {}
    site_suffix = f"_{site_name}" if site_name else ""

    # Process each axis individually
    axis_names = ['X', 'Y', 'Z']
    for i, axis in enumerate(axis_names):
        data = acc_data[i, :]
        prefix = f"{axis}{site_suffix}_acc"

        if data.size == 0:
            # Handle empty data gracefully
            _add_nan_features(features, prefix)
            continue

        # Statistical features
        features[f"{prefix}_Mean"] = float(np.mean(data))
        features[f"{prefix}_Std"] = float(np.std(data, ddof=1)) if data.size > 1 else np.nan
        features[f"{prefix}_Max"] = float(np.max(data))
        features[f"{prefix}_Min"] = float(np.min(data))
        features[f"{prefix}_Range"] = float(np.max(data) - np.min(data))
        features[f"{prefix}_RMS"] = float(np.sqrt(np.mean(data ** 2)))
        features[f"{prefix}_Energy"] = float(np.sum(data ** 2))
        features[f"{prefix}_IQR"] = float(np.percentile(data, 75) - np.percentile(data, 25))

        # Shape features
        features[f"{prefix}_Skewness"] = float(skew(data)) if data.size > 2 else np.nan
        features[f"{prefix}_Kurtosis"] = float(kurtosis(data)) if data.size > 3 else np.nan

    # Compute magnitude features
    magnitude = np.sqrt(np.sum(acc_data ** 2, axis=0))
    mag_prefix = f"Magnitude{site_suffix}_acc"

    if magnitude.size == 0:
        _add_nan_features(features, mag_prefix)
    else:
        # Statistical features for magnitude
        features[f"{mag_prefix}_Mean"] = float(np.mean(magnitude))
        features[f"{mag_prefix}_Std"] = float(np.std(magnitude, ddof=1)) if magnitude.size > 1 else np.nan
        features[f"{mag_prefix}_Max"] = float(np.max(magnitude))
        features[f"{mag_prefix}_Min"] = float(np.min(magnitude))
        features[f"{mag_prefix}_Range"] = float(np.max(magnitude) - np.min(magnitude))
        features[f"{mag_prefix}_RMS"] = float(np.sqrt(np.mean(magnitude ** 2)))
        features[f"{mag_prefix}_Energy"] = float(np.sum(magnitude ** 2))
        features[f"{mag_prefix}_IQR"] = float(np.percentile(magnitude, 75) - np.percentile(magnitude, 25))

        # Shape features for magnitude
        features[f"{mag_prefix}_Skewness"] = float(skew(magnitude)) if magnitude.size > 2 else np.nan
        features[f"{mag_prefix}_Kurtosis"] = float(kurtosis(magnitude)) if magnitude.size > 3 else np.nan

    # Optional: Add duration feature
    duration_s = acc_data.shape[1] / fs
    features[f"Duration{site_suffix}_acc"] = float(duration_s)

    return features


def _add_nan_features(features: Dict[str, float], prefix: str) -> None:
    """Helper to add NaN features when data is missing/empty."""
    feature_names = [
        "Mean", "Std", "Max", "Min", "Range", "RMS",
        "Energy", "IQR", "Skewness", "Kurtosis"
    ]
    for name in feature_names:
        features[f"{prefix}_{name}"] = np.nan


def compute_multi_axis_acc_features(
        acc_data: np.ndarray,
        fs: float,
        site_name: Optional[str] = None,
        include_cross_axis: bool = False
) -> Dict[str, float]:
    """
    Extended 3-axis accelerometer features including cross-axis relationships.

    Parameters:
    -----------
    acc_data : np.ndarray
        Accelerometer data shaped (3, n_samples)
    fs : float
        Sampling frequency in Hz
    site_name : str, optional
        Sensor site name for feature naming
    include_cross_axis : bool, default False
        Whether to include cross-axis correlation features

    Returns:
    --------
    Dict[str, float]
        Dictionary with basic 3-axis features plus optional cross-axis features
    """
    # Get basic features
    features = compute_3axis_acc_features(acc_data, fs, site_name)

    if include_cross_axis and acc_data.shape[1] > 1:
        site_suffix = f"_{site_name}" if site_name else ""

        # Cross-axis correlations
        x, y, z = acc_data[0, :], acc_data[1, :], acc_data[2, :]

        try:
            features[f"XY_corr{site_suffix}_acc"] = float(np.corrcoef(x, y)[0, 1])
            features[f"XZ_corr{site_suffix}_acc"] = float(np.corrcoef(x, z)[0, 1])
            features[f"YZ_corr{site_suffix}_acc"] = float(np.corrcoef(y, z)[0, 1])
        except:
            # Handle cases where correlation can't be computed
            features[f"XY_corr{site_suffix}_acc"] = np.nan
            features[f"XZ_corr{site_suffix}_acc"] = np.nan
            features[f"YZ_corr{site_suffix}_acc"] = np.nan

    return features


def compute_acc_features_for_window(
        window: Dict,
        fs_imu: float,
        site_name: Optional[str] = None,
        extended: bool = False
) -> Dict[str, float]:
    """
    Convenience function to extract accelerometer features from a window dict.
    This is designed to integrate with your library's windowing system.

    Parameters:
    -----------
    window : Dict
        Window dictionary containing 'acc' key with shape (3, n_samples)
    fs_imu : float
        IMU sampling frequency
    site_name : str, optional
        Sensor site name for feature naming
    extended : bool, default False
        Whether to include extended features (cross-axis correlations)

    Returns:
    --------
    Dict[str, float]
        Dictionary of accelerometer features
    """
    if "acc" not in window or window["acc"] is None:
        return {}

    acc_data = np.asarray(window["acc"])

    if extended:
        return compute_multi_axis_acc_features(
            acc_data, fs_imu, site_name, include_cross_axis=True
        )
    else:
        return compute_3axis_acc_features(acc_data, fs_imu, site_name)

