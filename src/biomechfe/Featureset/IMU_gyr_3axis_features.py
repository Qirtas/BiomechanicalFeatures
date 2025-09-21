"""
3-axis gyroscope (angular velocity) feature extraction for biomechanical analysis.
Includes basic statistical features, fatigue-specific features, coordination metrics,
and advanced rotational analysis.
"""

import numpy as np
from scipy.stats import skew, kurtosis, linregress
from scipy.signal import savgol_filter
from typing import Dict, Optional


def compute_sample_entropy(signal: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """
    Compute sample entropy for a time series to measure signal regularity.

    Parameters:
    -----------
    signal : np.ndarray
        Input signal
    m : int, default 2
        Embedding dimension
    r : float, default 0.2
        Tolerance (typically 0.1-0.25 of signal std)

    Returns:
    --------
    float
        Sample entropy value (higher = more irregular)
    """
    if len(signal) < m + 2:
        return np.nan

    # Normalize r if it's a proportion of std
    if r < 1:
        r = r * np.std(signal)

    def create_vectors(data, m):
        vectors = []
        for i in range(len(data) - m + 1):
            vectors.append(data[i:i + m])
        return np.array(vectors)

    def count_matches(vectors, r):
        N = len(vectors)
        B = 0
        for i in range(N - 1):
            distances = np.max(np.abs(vectors[i] - vectors[i + 1:]), axis=1)
            B += np.sum(distances < r)
        return B / ((N - 1) * (N - 1))

    try:
        vectors_m = create_vectors(signal, m)
        vectors_m1 = create_vectors(signal, m + 1)

        B_m = count_matches(vectors_m, r)
        B_m1 = count_matches(vectors_m1, r)

        if B_m == 0 or B_m1 == 0:
            return np.nan

        return -np.log(B_m1 / B_m)
    except:
        return np.nan


def compute_trend_features(signal: np.ndarray) -> Dict[str, float]:
    """
    Compute trend features to capture progressive changes (fatigue indicators).

    Parameters:
    -----------
    signal : np.ndarray
        Input signal

    Returns:
    --------
    Dict[str, float]
        Dictionary of trend features
    """
    features = {}

    if len(signal) < 3:
        features["LinearTrend_Slope"] = np.nan
        features["LinearTrend_R2"] = np.nan
        features["FirstHalf_Mean"] = np.nan
        features["SecondHalf_Mean"] = np.nan
        features["MeanRatio"] = np.nan
        features["ProgressiveRatio"] = np.nan
        return features

    # Linear trend analysis
    try:
        x = np.arange(len(signal))
        slope, intercept, r_value, p_value, std_err = linregress(x, signal)
        features["LinearTrend_Slope"] = slope
        features["LinearTrend_R2"] = r_value ** 2
    except:
        features["LinearTrend_Slope"] = np.nan
        features["LinearTrend_R2"] = np.nan

    # First half vs second half comparison
    half_idx = len(signal) // 2
    first_half = signal[:half_idx]
    second_half = signal[half_idx:]

    if len(first_half) > 0 and len(second_half) > 0:
        first_mean = np.mean(first_half)
        second_mean = np.mean(second_half)

        features["FirstHalf_Mean"] = first_mean
        features["SecondHalf_Mean"] = second_mean

        # Ratio of second half to first half (captures progressive changes)
        if abs(first_mean) > 1e-10:
            features["MeanRatio"] = second_mean / first_mean
        else:
            features["MeanRatio"] = np.nan

        # Progressive ratio: divide signal into quarters
        quarter_size = len(signal) // 4
        if quarter_size > 0:
            q1 = np.mean(signal[:quarter_size])
            q4 = np.mean(signal[-quarter_size:])

            if abs(q1) > 1e-10:
                features["ProgressiveRatio"] = q4 / q1
            else:
                features["ProgressiveRatio"] = np.nan
        else:
            features["ProgressiveRatio"] = np.nan
    else:
        features["FirstHalf_Mean"] = np.nan
        features["SecondHalf_Mean"] = np.nan
        features["MeanRatio"] = np.nan
        features["ProgressiveRatio"] = np.nan

    return features


def compute_angular_displacement(angular_velocity: np.ndarray, sampling_rate: float) -> np.ndarray:
    """
    Compute angular displacement by integrating angular velocity.

    Parameters:
    -----------
    angular_velocity : np.ndarray
        Angular velocity signal
    sampling_rate : float
        Sampling rate in Hz

    Returns:
    --------
    np.ndarray
        Angular displacement signal
    """
    dt = 1.0 / sampling_rate
    angular_displacement = np.cumsum(angular_velocity) * dt

    # Remove drift using detrending
    if len(angular_displacement) > 3:
        angular_displacement = angular_displacement - np.mean(angular_displacement)

    return angular_displacement


def compute_angular_jerk(angular_velocity: np.ndarray, sampling_rate: float) -> np.ndarray:
    """
    Compute angular jerk (rate of change of angular acceleration).

    Parameters:
    -----------
    angular_velocity : np.ndarray
        Angular velocity signal
    sampling_rate : float
        Sampling rate in Hz

    Returns:
    --------
    np.ndarray
        Angular jerk signal
    """
    if len(angular_velocity) < 3:
        return np.array([])

    dt = 1.0 / sampling_rate

    # First compute angular acceleration
    angular_acceleration = np.diff(angular_velocity) / dt

    # Then compute angular jerk
    angular_jerk = np.diff(angular_acceleration) / dt

    # Optional smoothing to reduce noise
    if len(angular_jerk) > 5:
        try:
            window_length = min(11, len(angular_jerk) - (len(angular_jerk) % 2) - 1)
            if window_length > 2:
                angular_jerk = savgol_filter(angular_jerk, window_length, 2)
        except:
            pass

    return angular_jerk


def compute_rotational_energy_metrics(
        angular_velocity: np.ndarray,
        angular_displacement: np.ndarray,
        sampling_rate: float
) -> Dict[str, float]:
    """
    Compute rotational energy metrics.

    Parameters:
    -----------
    angular_velocity : np.ndarray
        Angular velocity signal
    angular_displacement : np.ndarray
        Angular displacement signal
    sampling_rate : float
        Sampling rate in Hz

    Returns:
    --------
    Dict[str, float]
        Dictionary of rotational energy metrics
    """
    metrics = {}

    if len(angular_velocity) < 2 or len(angular_displacement) < 2:
        metrics["RotationalKineticEnergy_Mean"] = np.nan
        metrics["RotationalWork_Total"] = np.nan
        metrics["RotationalPower_Mean"] = np.nan
        metrics["RotationalPower_Peak"] = np.nan
        return metrics

    # Rotational kinetic energy (assuming unit moment of inertia)
    rotational_kinetic_energy = 0.5 * angular_velocity ** 2

    # Rotational work approximation using changes in kinetic energy
    rotational_work = np.abs(np.diff(rotational_kinetic_energy))

    # Rotational power
    dt = 1.0 / sampling_rate
    rotational_power = rotational_work / dt

    # Calculate metrics
    metrics["RotationalKineticEnergy_Mean"] = np.mean(rotational_kinetic_energy)
    metrics["RotationalWork_Total"] = np.sum(rotational_work)
    metrics["RotationalPower_Mean"] = np.mean(rotational_power) if len(rotational_power) > 0 else np.nan
    metrics["RotationalPower_Peak"] = np.max(rotational_power) if len(rotational_power) > 0 else np.nan

    return metrics


def compute_coordination_metrics(
        x_velocity: np.ndarray,
        y_velocity: np.ndarray,
        z_velocity: np.ndarray
) -> Dict[str, float]:
    """
    Compute coordination metrics between different axes.

    Parameters:
    -----------
    x_velocity : np.ndarray
        Angular velocity in X axis
    y_velocity : np.ndarray
        Angular velocity in Y axis
    z_velocity : np.ndarray
        Angular velocity in Z axis

    Returns:
    --------
    Dict[str, float]
        Dictionary of coordination metrics
    """
    metrics = {}

    if len(x_velocity) < 2 or len(y_velocity) < 2 or len(z_velocity) < 2:
        metrics["XY_Correlation"] = np.nan
        metrics["XZ_Correlation"] = np.nan
        metrics["YZ_Correlation"] = np.nan
        metrics["PrincipalAxis_Ratio"] = np.nan
        metrics["RotationalSmoothness"] = np.nan
        return metrics

    # Correlation between axes
    try:
        metrics["XY_Correlation"] = np.corrcoef(x_velocity, y_velocity)[0, 1]
        metrics["XZ_Correlation"] = np.corrcoef(x_velocity, z_velocity)[0, 1]
        metrics["YZ_Correlation"] = np.corrcoef(y_velocity, z_velocity)[0, 1]
    except:
        metrics["XY_Correlation"] = np.nan
        metrics["XZ_Correlation"] = np.nan
        metrics["YZ_Correlation"] = np.nan

    # Principal component analysis for rotation plane
    try:
        data = np.column_stack((x_velocity, y_velocity, z_velocity))
        data_centered = data - np.mean(data, axis=0)
        cov_matrix = np.cov(data_centered, rowvar=False)
        eigenvalues = np.linalg.eigvals(cov_matrix)
        eigenvalues = np.sort(np.abs(eigenvalues))[::-1]

        if eigenvalues[1] > 1e-10:
            metrics["PrincipalAxis_Ratio"] = eigenvalues[0] / eigenvalues[1]
        else:
            metrics["PrincipalAxis_Ratio"] = np.nan
    except:
        metrics["PrincipalAxis_Ratio"] = np.nan

    # Rotational smoothness
    try:
        magnitude = np.sqrt(x_velocity ** 2 + y_velocity ** 2 + z_velocity ** 2)
        jerk = np.diff(np.diff(magnitude))
        if len(jerk) > 0 and np.sum(magnitude ** 2) > 1e-10:
            metrics["RotationalSmoothness"] = -np.log(np.mean(jerk ** 2) / np.mean(magnitude ** 2))
        else:
            metrics["RotationalSmoothness"] = np.nan
    except:
        metrics["RotationalSmoothness"] = np.nan

    return metrics


def compute_gyr_features(
        gyr_data: np.ndarray,
        fs: float,
        site_name: Optional[str] = None
) -> Dict[str, float]:
    """
    Compute comprehensive 3-axis gyroscope features.

    Parameters:
    -----------
    gyr_data : np.ndarray
        Gyroscope data shaped (3, n_samples) where rows are [X, Y, Z]
    fs : float
        Sampling frequency in Hz
    site_name : str, optional
        Name of the sensor site for feature naming

    Returns:
    --------
    Dict[str, float]
        Dictionary containing gyroscope features
    """
    if gyr_data.shape[0] != 3:
        raise ValueError(f"Expected gyr_data shape (3, n_samples), got {gyr_data.shape}")

    features = {}
    site_suffix = f"_{site_name}" if site_name else ""
    axis_names = ['X', 'Y', 'Z']

    # Store data for coordination metrics
    angular_velocities = {}

    # Process each axis individually
    for i, axis in enumerate(axis_names):
        data = gyr_data[i, :]
        prefix = f"{axis}{site_suffix}_gyr"
        angular_velocities[axis] = data

        if data.size == 0:
            _add_nan_gyr_features(features, prefix)
            continue

        # Basic statistical features
        features[f"{prefix}_Mean"] = float(np.mean(data))
        features[f"{prefix}_Std"] = float(np.std(data, ddof=1)) if data.size > 1 else np.nan
        features[f"{prefix}_Max"] = float(np.max(data))
        features[f"{prefix}_Min"] = float(np.min(data))
        features[f"{prefix}_Range"] = float(np.max(data) - np.min(data))
        features[f"{prefix}_RMS"] = float(np.sqrt(np.mean(data ** 2)))
        features[f"{prefix}_Energy"] = float(np.sum(data ** 2))
        features[f"{prefix}_IQR"] = float(np.percentile(data, 75) - np.percentile(data, 25))
        features[f"{prefix}_Skewness"] = float(skew(data)) if data.size > 2 else np.nan
        features[f"{prefix}_Kurtosis"] = float(kurtosis(data)) if data.size > 3 else np.nan

        # Fatigue-specific features
        try:
            features[f"{prefix}_SampleEntropy"] = compute_sample_entropy(data, m=2, r=0.2)
        except:
            features[f"{prefix}_SampleEntropy"] = np.nan

        # Trend analysis
        try:
            trend_features = compute_trend_features(data)
            features[f"{prefix}_LinearTrend_Slope"] = trend_features["LinearTrend_Slope"]
            features[f"{prefix}_LinearTrend_R2"] = trend_features["LinearTrend_R2"]
            features[f"{prefix}_MeanRatio"] = trend_features["MeanRatio"]
            features[f"{prefix}_ProgressiveRatio"] = trend_features["ProgressiveRatio"]
        except:
            features[f"{prefix}_LinearTrend_Slope"] = np.nan
            features[f"{prefix}_LinearTrend_R2"] = np.nan
            features[f"{prefix}_MeanRatio"] = np.nan
            features[f"{prefix}_ProgressiveRatio"] = np.nan

        # Angular displacement features
        try:
            ang_disp = compute_angular_displacement(data, fs)
            features[f"{prefix}_AngDisp_Mean"] = float(np.mean(ang_disp))
            features[f"{prefix}_AngDisp_Range"] = float(np.max(ang_disp) - np.min(ang_disp))
            features[f"{prefix}_AngDisp_Total"] = float(np.sum(np.abs(np.diff(ang_disp))))
        except:
            features[f"{prefix}_AngDisp_Mean"] = np.nan
            features[f"{prefix}_AngDisp_Range"] = np.nan
            features[f"{prefix}_AngDisp_Total"] = np.nan

        # Angular jerk features
        try:
            ang_jerk = compute_angular_jerk(data, fs)
            if len(ang_jerk) > 0:
                features[f"{prefix}_AngJerk_Mean"] = float(np.mean(ang_jerk))
                features[f"{prefix}_AngJerk_Std"] = float(np.std(ang_jerk, ddof=1)) if len(ang_jerk) > 1 else np.nan
                features[f"{prefix}_AngJerk_RMS"] = float(np.sqrt(np.mean(ang_jerk ** 2)))
                features[f"{prefix}_AngJerk_Energy"] = float(np.sum(ang_jerk ** 2))
            else:
                features[f"{prefix}_AngJerk_Mean"] = np.nan
                features[f"{prefix}_AngJerk_Std"] = np.nan
                features[f"{prefix}_AngJerk_RMS"] = np.nan
                features[f"{prefix}_AngJerk_Energy"] = np.nan
        except:
            features[f"{prefix}_AngJerk_Mean"] = np.nan
            features[f"{prefix}_AngJerk_Std"] = np.nan
            features[f"{prefix}_AngJerk_RMS"] = np.nan
            features[f"{prefix}_AngJerk_Energy"] = np.nan

    # Magnitude features
    if all(gyr_data[i, :].size > 0 for i in range(3)):
        x, y, z = gyr_data[0, :], gyr_data[1, :], gyr_data[2, :]
        magnitude = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        mag_prefix = f"Magnitude{site_suffix}_gyr"

        # Basic statistical features for magnitude
        features[f"{mag_prefix}_Mean"] = float(np.mean(magnitude))
        features[f"{mag_prefix}_Std"] = float(np.std(magnitude, ddof=1)) if magnitude.size > 1 else np.nan
        features[f"{mag_prefix}_Max"] = float(np.max(magnitude))
        features[f"{mag_prefix}_Min"] = float(np.min(magnitude))
        features[f"{mag_prefix}_Range"] = float(np.max(magnitude) - np.min(magnitude))
        features[f"{mag_prefix}_RMS"] = float(np.sqrt(np.mean(magnitude ** 2)))
        features[f"{mag_prefix}_Energy"] = float(np.sum(magnitude ** 2))
        features[f"{mag_prefix}_IQR"] = float(np.percentile(magnitude, 75) - np.percentile(magnitude, 25))
        features[f"{mag_prefix}_Skewness"] = float(skew(magnitude)) if magnitude.size > 2 else np.nan
        features[f"{mag_prefix}_Kurtosis"] = float(kurtosis(magnitude)) if magnitude.size > 3 else np.nan

        # Sample entropy and trend for magnitude
        try:
            features[f"{mag_prefix}_SampleEntropy"] = compute_sample_entropy(magnitude, m=2, r=0.2)
        except:
            features[f"{mag_prefix}_SampleEntropy"] = np.nan

        try:
            trend_features = compute_trend_features(magnitude)
            features[f"{mag_prefix}_LinearTrend_Slope"] = trend_features["LinearTrend_Slope"]
            features[f"{mag_prefix}_LinearTrend_R2"] = trend_features["LinearTrend_R2"]
            features[f"{mag_prefix}_MeanRatio"] = trend_features["MeanRatio"]
            features[f"{mag_prefix}_ProgressiveRatio"] = trend_features["ProgressiveRatio"]
        except:
            features[f"{mag_prefix}_LinearTrend_Slope"] = np.nan
            features[f"{mag_prefix}_LinearTrend_R2"] = np.nan
            features[f"{mag_prefix}_MeanRatio"] = np.nan
            features[f"{mag_prefix}_ProgressiveRatio"] = np.nan

        # Angular jerk for magnitude
        try:
            mag_jerk = compute_angular_jerk(magnitude, fs)
            if len(mag_jerk) > 0:
                features[f"{mag_prefix}_AngJerk_Mean"] = float(np.mean(mag_jerk))
                features[f"{mag_prefix}_AngJerk_Std"] = float(np.std(mag_jerk, ddof=1)) if len(mag_jerk) > 1 else np.nan
                features[f"{mag_prefix}_AngJerk_RMS"] = float(np.sqrt(np.mean(mag_jerk ** 2)))
                features[f"{mag_prefix}_AngJerk_Energy"] = float(np.sum(mag_jerk ** 2))
            else:
                features[f"{mag_prefix}_AngJerk_Mean"] = np.nan
                features[f"{mag_prefix}_AngJerk_Std"] = np.nan
                features[f"{mag_prefix}_AngJerk_RMS"] = np.nan
                features[f"{mag_prefix}_AngJerk_Energy"] = np.nan
        except:
            features[f"{mag_prefix}_AngJerk_Mean"] = np.nan
            features[f"{mag_prefix}_AngJerk_Std"] = np.nan
            features[f"{mag_prefix}_AngJerk_RMS"] = np.nan
            features[f"{mag_prefix}_AngJerk_Energy"] = np.nan

        # Rotational energy metrics
        try:
            mag_disp = compute_angular_displacement(magnitude, fs)
            energy_metrics = compute_rotational_energy_metrics(magnitude, mag_disp, fs)
            features[f"{mag_prefix}_RotKineticEnergy_Mean"] = energy_metrics["RotationalKineticEnergy_Mean"]
            features[f"{mag_prefix}_RotWork_Total"] = energy_metrics["RotationalWork_Total"]
            features[f"{mag_prefix}_RotPower_Mean"] = energy_metrics["RotationalPower_Mean"]
            features[f"{mag_prefix}_RotPower_Peak"] = energy_metrics["RotationalPower_Peak"]
        except:
            features[f"{mag_prefix}_RotKineticEnergy_Mean"] = np.nan
            features[f"{mag_prefix}_RotWork_Total"] = np.nan
            features[f"{mag_prefix}_RotPower_Mean"] = np.nan
            features[f"{mag_prefix}_RotPower_Peak"] = np.nan

        # Coordination metrics
        try:
            coord_metrics = compute_coordination_metrics(
                angular_velocities['X'], angular_velocities['Y'], angular_velocities['Z']
            )
            coord_prefix = f"Coord{site_suffix}_gyr"
            features[f"{coord_prefix}_XY_Correlation"] = coord_metrics["XY_Correlation"]
            features[f"{coord_prefix}_XZ_Correlation"] = coord_metrics["XZ_Correlation"]
            features[f"{coord_prefix}_YZ_Correlation"] = coord_metrics["YZ_Correlation"]
            features[f"{coord_prefix}_PrincipalAxis_Ratio"] = coord_metrics["PrincipalAxis_Ratio"]
            features[f"{coord_prefix}_RotationalSmoothness"] = coord_metrics["RotationalSmoothness"]
        except:
            coord_prefix = f"Coord{site_suffix}_gyr"
            features[f"{coord_prefix}_XY_Correlation"] = np.nan
            features[f"{coord_prefix}_XZ_Correlation"] = np.nan
            features[f"{coord_prefix}_YZ_Correlation"] = np.nan
            features[f"{coord_prefix}_PrincipalAxis_Ratio"] = np.nan
            features[f"{coord_prefix}_RotationalSmoothness"] = np.nan

    return features


def _add_nan_gyr_features(features: Dict[str, float], prefix: str) -> None:
    """Helper to add NaN features when data is missing/empty."""
    feature_names = [
        "Mean", "Std", "Max", "Min", "Range", "RMS", "Energy", "IQR",
        "Skewness", "Kurtosis", "SampleEntropy", "LinearTrend_Slope",
        "LinearTrend_R2", "MeanRatio", "ProgressiveRatio", "AngDisp_Mean",
        "AngDisp_Range", "AngDisp_Total", "AngJerk_Mean", "AngJerk_Std",
        "AngJerk_RMS", "AngJerk_Energy"
    ]
    for name in feature_names:
        features[f"{prefix}_{name}"] = np.nan


def compute_gyr_features_for_window(
        window: Dict,
        fs_imu: float,
        site_name: Optional[str] = None
) -> Dict[str, float]:
    """
    Convenience function to extract gyroscope features from a window dict.

    Parameters:
    -----------
    window : Dict
        Window dictionary containing 'gyr' key with shape (3, n_samples)
    fs_imu : float
        IMU sampling frequency
    site_name : str, optional
        Sensor site name for feature naming

    Returns:
    --------
    Dict[str, float]
        Dictionary of gyroscope features
    """
    if "gyr" not in window or window["gyr"] is None:
        return {}

    gyr_data = np.asarray(window["gyr"])

    return compute_gyr_features(gyr_data, fs_imu, site_name)
