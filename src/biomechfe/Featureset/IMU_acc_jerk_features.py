"""
Jerk feature extraction from 3-axis accelerometer data for biomechanical analysis.
Jerk is the time derivative of acceleration and provides insights into movement smoothness.
"""

import numpy as np
from scipy.signal import filtfilt, butter
from scipy.stats import skew, kurtosis
from typing import Dict, Optional, Tuple


def compute_jerk_features(
        acc_data: np.ndarray,
        fs: float,
        site_name: Optional[str] = None,
        filter_params: Optional[Dict] = None
) -> Dict[str, float]:
    """
    Compute comprehensive jerk features from 3-axis accelerometer data.

    Parameters:
    -----------
    acc_data : np.ndarray
        Accelerometer data shaped (3, n_samples) where rows are [X, Y, Z]
    fs : float
        Sampling frequency in Hz
    site_name : str, optional
        Name of the sensor site (e.g., 'Shoulder', 'Wrist') for feature naming.
        If None, uses generic naming.
    filter_params : dict, optional
        Filter parameters: {'order': int, 'cutoff': float}
        Default: {'order': 2, 'cutoff': 0.1} (relative to Nyquist)

    Returns:
    --------
    Dict[str, float]
        Dictionary containing jerk features:
        - Individual axis jerk features: jerkX_{site}_mean, jerkY_{site}_std, etc.
        - Magnitude jerk features: jerkMag_{site}_mean, etc.
        - Normalized jerk score (movement smoothness measure)
    """
    if acc_data.shape[0] != 3:
        raise ValueError(f"Expected acc_data shape (3, n_samples), got {acc_data.shape}")

    if acc_data.shape[1] < 2:
        # Need at least 2 samples to compute jerk (derivative)
        return _create_empty_jerk_features(site_name)

    # Default filter parameters
    if filter_params is None:
        filter_params = {'order': 2, 'cutoff': 0.1}

    features = {}
    site_suffix = f"_{site_name}" if site_name else ""
    dt = 1.0 / fs

    # Extract individual axes
    ax, ay, az = acc_data[0, :], acc_data[1, :], acc_data[2, :]

    # Apply smoothing filter before differentiation
    try:
        b, a = butter(filter_params['order'], filter_params['cutoff'])
        ax_smooth = filtfilt(b, a, ax)
        ay_smooth = filtfilt(b, a, ay)
        az_smooth = filtfilt(b, a, az)
    except Exception:
        # If filtering fails (e.g., signal too short), use original data
        ax_smooth, ay_smooth, az_smooth = ax, ay, az

    # Compute jerk (first derivative of acceleration)
    jerk_x = np.diff(ax_smooth) / dt
    jerk_y = np.diff(ay_smooth) / dt
    jerk_z = np.diff(az_smooth) / dt

    # Compute jerk magnitude
    jerk_magnitude = np.sqrt(jerk_x ** 2 + jerk_y ** 2 + jerk_z ** 2)

    # Compute features for each jerk component
    jerk_components = {
        'jerkX': jerk_x,
        'jerkY': jerk_y,
        'jerkZ': jerk_z,
        'jerkMag': jerk_magnitude
    }

    for component_name, jerk_signal in jerk_components.items():
        prefix = f"{component_name}{site_suffix}"
        component_features = _compute_jerk_statistics(jerk_signal, prefix)
        features.update(component_features)

    # Compute normalized jerk score (measure of movement smoothness)
    movement_duration = acc_data.shape[1] * dt
    jerk_squared_sum = np.sum(jerk_magnitude ** 2)
    normalized_jerk = np.sqrt(jerk_squared_sum * (movement_duration ** 5) / 2)
    features[f"normalizedJerk{site_suffix}"] = float(normalized_jerk)

    return features


def _compute_jerk_statistics(jerk_signal: np.ndarray, prefix: str) -> Dict[str, float]:
    """
    Calculate comprehensive statistical features for a jerk signal.

    Parameters:
    -----------
    jerk_signal : np.ndarray
        1D array of jerk values
    prefix : str
        Prefix for feature names (e.g., 'jerkX_Shoulder', 'jerkMag')

    Returns:
    --------
    Dict[str, float]
        Dictionary of statistical features
    """
    features = {}

    if jerk_signal.size == 0:
        # Handle empty signal
        feature_names = ['mean', 'median', 'min', 'max', 'skew', 'kurtosis',
                         'std', 'rms', 'range', 'energy', 'iqr']
        for name in feature_names:
            features[f"{prefix}_{name}"] = np.nan
        return features

    # Basic statistical features
    features[f"{prefix}_mean"] = float(np.mean(jerk_signal))
    features[f"{prefix}_median"] = float(np.median(jerk_signal))
    features[f"{prefix}_min"] = float(np.min(jerk_signal))
    features[f"{prefix}_max"] = float(np.max(jerk_signal))
    features[f"{prefix}_range"] = float(np.max(jerk_signal) - np.min(jerk_signal))

    # Standard deviation (sample std)
    features[f"{prefix}_std"] = float(np.std(jerk_signal, ddof=1)) if jerk_signal.size > 1 else np.nan

    # RMS (Root Mean Square)
    features[f"{prefix}_rms"] = float(np.sqrt(np.mean(jerk_signal ** 2)))

    # Energy
    features[f"{prefix}_energy"] = float(np.sum(jerk_signal ** 2))

    # Interquartile Range
    features[f"{prefix}_iqr"] = float(np.percentile(jerk_signal, 75) - np.percentile(jerk_signal, 25))

    # Shape features
    features[f"{prefix}_skew"] = float(skew(jerk_signal)) if jerk_signal.size > 2 else np.nan
    features[f"{prefix}_kurtosis"] = float(kurtosis(jerk_signal)) if jerk_signal.size > 3 else np.nan

    return features


def _create_empty_jerk_features(site_name: Optional[str]) -> Dict[str, float]:
    """Create a dictionary of NaN jerk features for cases with insufficient data."""
    site_suffix = f"_{site_name}" if site_name else ""
    features = {}

    # Define all jerk components
    components = ['jerkX', 'jerkY', 'jerkZ', 'jerkMag']
    feature_names = ['mean', 'median', 'min', 'max', 'skew', 'kurtosis',
                     'std', 'rms', 'range', 'energy', 'iqr']

    # Create NaN features for all components
    for component in components:
        prefix = f"{component}{site_suffix}"
        for name in feature_names:
            features[f"{prefix}_{name}"] = np.nan

    # Add normalized jerk
    features[f"normalizedJerk{site_suffix}"] = np.nan

    return features


def compute_jerk_features_for_window(
        window: Dict,
        fs_imu: float,
        site_name: Optional[str] = None,
        filter_params: Optional[Dict] = None
) -> Dict[str, float]:
    """
    Convenience function to extract jerk features from a window dict.
    This is designed to integrate with your library's windowing system.

    Parameters:
    -----------
    window : Dict
        Window dictionary containing 'acc' key with shape (3, n_samples)
    fs_imu : float
        IMU sampling frequency
    site_name : str, optional
        Sensor site name for feature naming
    filter_params : dict, optional
        Filter parameters for smoothing before differentiation

    Returns:
    --------
    Dict[str, float]
        Dictionary of jerk features
    """
    if "acc" not in window or window["acc"] is None:
        return {}

    acc_data = np.asarray(window["acc"])

    return compute_jerk_features(
        acc_data, fs_imu, site_name, filter_params
    )


def compute_advanced_jerk_features(
        acc_data: np.ndarray,
        fs: float,
        site_name: Optional[str] = None,
        include_derivatives: bool = False
) -> Dict[str, float]:
    """
    Compute advanced jerk features including higher-order derivatives.

    Parameters:
    -----------
    acc_data : np.ndarray
        Accelerometer data shaped (3, n_samples)
    fs : float
        Sampling frequency in Hz
    site_name : str, optional
        Sensor site name for feature naming
    include_derivatives : bool, default False
        Whether to include snap (jounce) - second derivative of jerk

    Returns:
    --------
    Dict[str, float]
        Dictionary with standard jerk features plus optional higher derivatives
    """
    # Get standard jerk features
    features = compute_jerk_features(acc_data, fs, site_name)

    if include_derivatives and acc_data.shape[1] > 3:
        site_suffix = f"_{site_name}" if site_name else ""
        dt = 1.0 / fs

        # Compute jerk first
        ax, ay, az = acc_data[0, :], acc_data[1, :], acc_data[2, :]
        jerk_x = np.diff(ax) / dt
        jerk_y = np.diff(ay) / dt
        jerk_z = np.diff(az) / dt

        # Compute snap (second derivative of jerk)
        if len(jerk_x) > 1:  # Need at least 2 jerk samples
            snap_x = np.diff(jerk_x) / dt
            snap_y = np.diff(jerk_y) / dt
            snap_z = np.diff(jerk_z) / dt
            snap_magnitude = np.sqrt(snap_x ** 2 + snap_y ** 2 + snap_z ** 2)

            # Add snap features
            features[f"snapMag{site_suffix}_mean"] = float(np.mean(snap_magnitude))
            features[f"snapMag{site_suffix}_std"] = float(
                np.std(snap_magnitude, ddof=1)) if snap_magnitude.size > 1 else np.nan
            features[f"snapMag{site_suffix}_max"] = float(np.max(snap_magnitude))

    return features
