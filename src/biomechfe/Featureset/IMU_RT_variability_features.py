"""
Reaction Time (RT) Variability feature extraction from IMU signals.
Computes variability features from the difference between consecutive samples,
including fatigue-specific features, sample entropy, detrended fluctuation analysis,
and progressive trend analysis for biomechanical movement assessment.
"""

import numpy as np
from scipy.stats import skew, kurtosis, linregress
from typing import Dict, Optional


def compute_sample_entropy(signal: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """
    Compute sample entropy for a time series to quantify signal regularity/predictability.

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
        Sample entropy value (higher = more irregular/unpredictable)
    """
    if len(signal) < m + 2:
        return np.nan

    # Normalize r if it's a proportion of std
    if r < 1:
        r = r * np.std(signal)

    # Create embedding vectors
    def create_vectors(data, m):
        vectors = []
        for i in range(len(data) - m + 1):
            vectors.append(data[i:i + m])
        return np.array(vectors)

    # Count similar patterns
    def count_matches(vectors, r):
        N = len(vectors)
        B = 0
        for i in range(N - 1):
            # Calculate distances
            distances = np.max(np.abs(vectors[i] - vectors[i + 1:]), axis=1)
            # Count matches
            B += np.sum(distances < r)
        return B / ((N - 1) * (N - 1))

    # Compute for m and m+1
    try:
        vectors_m = create_vectors(signal, m)
        vectors_m1 = create_vectors(signal, m + 1)

        B_m = count_matches(vectors_m, r)
        B_m1 = count_matches(vectors_m1, r)

        # Avoid log(0)
        if B_m == 0 or B_m1 == 0:
            return np.nan

        return -np.log(B_m1 / B_m)
    except:
        return np.nan


def compute_dfa(signal: np.ndarray, scales: Optional[np.ndarray] = None, poly_order: int = 1) -> float:
    """
    Compute Detrended Fluctuation Analysis to assess long-range correlations.

    Parameters:
    -----------
    signal : np.ndarray
        Input signal
    scales : np.ndarray, optional
        Array of scales to use (default=None, will use 4 to N/10)
    poly_order : int, default 1
        Order of polynomial for detrending

    Returns:
    --------
    float
        DFA scaling exponent (alpha)
    """
    if len(signal) < 20:  # Need reasonable length for DFA
        return np.nan

    # Integrate the signal (cumulative sum of deviations from mean)
    y = np.cumsum(signal - np.mean(signal))

    # Define scales if not provided
    if scales is None:
        min_scale = 4
        max_scale = len(y) // 10
        if max_scale <= min_scale:
            max_scale = len(y) // 4
        if max_scale <= min_scale:
            return np.nan
        scales = np.unique(np.logspace(np.log10(min_scale), np.log10(max_scale), 10).astype(int))

    # Calculate fluctuation for each scale
    fluctuations = []
    for scale in scales:
        if scale >= len(y):
            continue

        # Number of segments
        n_segments = len(y) // scale

        # Skip if too few segments
        if n_segments < 1:
            continue

        # Reshape data into segments
        segments = len(y) - (len(y) % scale)
        y_reshaped = y[:segments].reshape((n_segments, scale))

        # Create time array for polynomial fit
        t = np.arange(scale)

        # Calculate local trend and fluctuation
        segment_fluctuations = []
        for segment in y_reshaped:
            # Fit polynomial
            p = np.polyfit(t, segment, poly_order)
            # Create trend
            trend = np.polyval(p, t)
            # Calculate fluctuation (RMS)
            segment_fluctuations.append(np.sqrt(np.mean((segment - trend) ** 2)))

        # Average fluctuation over all segments
        fluctuations.append(np.mean(segment_fluctuations))

    # Convert to numpy arrays
    scales = np.array(scales[:len(fluctuations)])
    fluctuations = np.array(fluctuations)

    if len(scales) < 2:
        return np.nan

    # Fit line to log-log plot to get scaling exponent
    try:
        log_scales = np.log(scales)
        log_fluct = np.log(fluctuations)

        # Linear regression to get slope (alpha)
        slope, _, _, _, _ = linregress(log_scales, log_fluct)
        return slope
    except:
        return np.nan


def compute_trend_features(signal: np.ndarray) -> Dict[str, float]:
    """
    Compute trend features to capture progressive changes within a movement.

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

        # Progressive ratio: divide signal into quarters and calculate ratio of means
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


def compute_rt_variability_features_for_signal(
        signal: np.ndarray,
        prefix: str,
        sampling_rate: float
) -> Dict[str, float]:
    """
    Compute RT variability features for one signal from the difference between consecutive samples.

    Parameters:
    -----------
    signal : np.ndarray
        1D signal array
    prefix : str
        Prefix for feature names (e.g., "X_Shoulder_acc_rtVar")
    sampling_rate : float
        Sampling rate in Hz

    Returns:
    --------
    Dict[str, float]
        Dictionary of RT variability features
    """
    features = {}

    if len(signal) < 2:
        # Not enough data to compute differences
        feature_names = [
            "Mean", "Std", "Max", "Min", "Range", "RMS", "Energy", "IQR",
            "Skewness", "Kurtosis", "SampleEntropy", "DFA_Alpha",
            "LinearTrend_Slope", "LinearTrend_R2", "MeanRatio", "ProgressiveRatio",
            "Raw_LinearTrend_Slope", "Raw_LinearTrend_R2", "Raw_MeanRatio", "Raw_ProgressiveRatio"
        ]
        for feature_name in feature_names:
            features[f"{prefix}_{feature_name}"] = np.nan
        return features

    # Compute differences (RT variability comes from consecutive sample differences)
    diff_data = np.diff(signal)

    # Basic statistical features on differences
    features[f"{prefix}_Mean"] = float(np.mean(diff_data))
    features[f"{prefix}_Std"] = float(np.std(diff_data, ddof=1)) if len(diff_data) > 1 else np.nan
    features[f"{prefix}_Max"] = float(np.max(diff_data))
    features[f"{prefix}_Min"] = float(np.min(diff_data))
    features[f"{prefix}_Range"] = float(np.max(diff_data) - np.min(diff_data))
    features[f"{prefix}_RMS"] = float(np.sqrt(np.mean(diff_data ** 2)))
    features[f"{prefix}_Energy"] = float(np.sum(diff_data ** 2))
    features[f"{prefix}_IQR"] = float(np.percentile(diff_data, 75) - np.percentile(diff_data, 25))

    # Shape features
    try:
        features[f"{prefix}_Skewness"] = float(skew(diff_data))
        features[f"{prefix}_Kurtosis"] = float(kurtosis(diff_data))
    except Exception:
        features[f"{prefix}_Skewness"] = np.nan
        features[f"{prefix}_Kurtosis"] = np.nan

    # Fatigue-specific features

    # 1. Sample Entropy - quantifies regularity/predictability
    try:
        # Use r=0.2*std as tolerance
        features[f"{prefix}_SampleEntropy"] = compute_sample_entropy(diff_data, m=2, r=0.2)
    except Exception:
        features[f"{prefix}_SampleEntropy"] = np.nan

    # 2. Detrended Fluctuation Analysis - long-range correlations
    try:
        features[f"{prefix}_DFA_Alpha"] = compute_dfa(diff_data)
    except Exception:
        features[f"{prefix}_DFA_Alpha"] = np.nan

    # 3. Trend Analysis - progressive changes
    try:
        # Compute trend features on both raw data and differences
        # Raw data trends capture overall movement pattern changes
        raw_trends = compute_trend_features(signal)
        # Difference trends capture changes in variability
        diff_trends = compute_trend_features(diff_data)

        # Add raw data trend features
        features[f"{prefix}_Raw_LinearTrend_Slope"] = raw_trends["LinearTrend_Slope"]
        features[f"{prefix}_Raw_LinearTrend_R2"] = raw_trends["LinearTrend_R2"]
        features[f"{prefix}_Raw_MeanRatio"] = raw_trends["MeanRatio"]
        features[f"{prefix}_Raw_ProgressiveRatio"] = raw_trends["ProgressiveRatio"]

        # Add difference trend features
        features[f"{prefix}_LinearTrend_Slope"] = diff_trends["LinearTrend_Slope"]
        features[f"{prefix}_LinearTrend_R2"] = diff_trends["LinearTrend_R2"]
        features[f"{prefix}_MeanRatio"] = diff_trends["MeanRatio"]
        features[f"{prefix}_ProgressiveRatio"] = diff_trends["ProgressiveRatio"]
    except Exception:
        features[f"{prefix}_Raw_LinearTrend_Slope"] = np.nan
        features[f"{prefix}_Raw_LinearTrend_R2"] = np.nan
        features[f"{prefix}_Raw_MeanRatio"] = np.nan
        features[f"{prefix}_Raw_ProgressiveRatio"] = np.nan
        features[f"{prefix}_LinearTrend_Slope"] = np.nan
        features[f"{prefix}_LinearTrend_R2"] = np.nan
        features[f"{prefix}_MeanRatio"] = np.nan
        features[f"{prefix}_ProgressiveRatio"] = np.nan

    return features


def compute_rt_variability_features_3axis(
        imu_data: np.ndarray,
        sampling_rate: float,
        sensor_type: str,
        site_name: Optional[str] = None
) -> Dict[str, float]:
    """
    Compute RT variability features for 3-axis IMU data (accelerometer or gyroscope).

    Parameters:
    -----------
    imu_data : np.ndarray
        IMU data shaped (3, n_samples) where rows are [X, Y, Z]
    sampling_rate : float
        Sampling rate in Hz
    sensor_type : str
        Type of sensor ('acc' for accelerometer, 'gyr' for gyroscope)
    site_name : str, optional
        Name of the sensor site for feature naming

    Returns:
    --------
    Dict[str, float]
        Dictionary of RT variability features
    """
    if imu_data.shape[0] != 3:
        raise ValueError(f"Expected imu_data shape (3, n_samples), got {imu_data.shape}")

    features = {}
    site_suffix = f"_{site_name}" if site_name else ""

    # Process each axis individually
    axis_names = ['X', 'Y', 'Z']
    for i, axis in enumerate(axis_names):
        data = imu_data[i, :]
        prefix = f"{axis}{site_suffix}_{sensor_type}_rtVar"

        if data.size == 0:
            continue

        # Compute RT variability features for this axis
        axis_features = compute_rt_variability_features_for_signal(
            data, prefix, sampling_rate
        )
        features.update(axis_features)

    # Compute magnitude features
    if all(imu_data[i, :].size > 0 for i in range(3)):
        x, y, z = imu_data[0, :], imu_data[1, :], imu_data[2, :]
        magnitude = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        mag_prefix = f"Magnitude{site_suffix}_{sensor_type}_rtVar"

        mag_features = compute_rt_variability_features_for_signal(
            magnitude, mag_prefix, sampling_rate
        )
        features.update(mag_features)

    return features


def compute_rt_variability_features_for_window(
        window: Dict,
        fs_imu: float,
        site_name: Optional[str] = None,
        include_acc: bool = True,
        include_gyr: bool = True
) -> Dict[str, float]:
    """
    Convenience function to extract RT variability features from a window dict.

    Parameters:
    -----------
    window : Dict
        Window dictionary containing 'acc' and/or 'gyr' keys
    fs_imu : float
        IMU sampling frequency
    site_name : str, optional
        Sensor site name for feature naming
    include_acc : bool, default True
        Whether to include accelerometer RT variability features
    include_gyr : bool, default True
        Whether to include gyroscope RT variability features

    Returns:
    --------
    Dict[str, float]
        Dictionary of RT variability features
    """
    features = {}

    # Process accelerometer data
    if include_acc and "acc" in window and window["acc"] is not None:
        acc_data = np.asarray(window["acc"])
        acc_features = compute_rt_variability_features_3axis(
            acc_data, fs_imu, "acc", site_name
        )
        features.update(acc_features)

    # Process gyroscope data
    if include_gyr and "gyr" in window and window["gyr"] is not None:
        gyr_data = np.asarray(window["gyr"])
        gyr_features = compute_rt_variability_features_3axis(
            gyr_data, fs_imu, "gyr", site_name
        )
        features.update(gyr_features)

    return features


def compute_emg_rt_variability_features_for_window(
        window: Dict,
        fs_emg: float,
        site_name: Optional[str] = None
) -> Dict[str, float]:
    """
    Convenience function to extract RT variability features from EMG data in a window dict.

    Parameters:
    -----------
    window : Dict
        Window dictionary containing 'emg' key with shape (n_channels, n_samples)
    fs_emg : float
        EMG sampling frequency
    site_name : str, optional
        EMG site name for feature naming

    Returns:
    --------
    Dict[str, float]
        Dictionary of EMG RT variability features
    """
    features = {}

    if "emg" not in window or window["emg"] is None:
        return features

    emg_data = np.asarray(window["emg"])  # Shape: (n_channels, n_samples)

    # Process each EMG channel
    for ch_idx in range(emg_data.shape[0]):
        channel_data = emg_data[ch_idx, :]

        if channel_data.size == 0:
            continue

        # Create prefix for this channel
        if site_name:
            prefix = f"ch{ch_idx}_{site_name}_emg_rtVar"
        else:
            prefix = f"ch{ch_idx}_emg_rtVar"

        # Compute RT variability features for this EMG channel
        channel_features = compute_rt_variability_features_for_signal(
            channel_data, prefix, fs_emg
        )
        features.update(channel_features)

    # Also compute features for the mean across all channels (if multiple channels)
    if emg_data.shape[0] > 1:
        mean_emg = np.mean(emg_data, axis=0)
        if site_name:
            mean_prefix = f"mean_{site_name}_emg_rtVar"
        else:
            mean_prefix = "mean_emg_rtVar"

        mean_features = compute_rt_variability_features_for_signal(
            mean_emg, mean_prefix, fs_emg
        )
        features.update(mean_features)

    return features


def compute_rt_variability_features_per_repetition(
        imu_data: np.ndarray,
        repetition_segments: list,
        sampling_rate: float,
        sensor_type: str,
        site_name: Optional[str] = None
) -> Dict[int, Dict[str, float]]:
    """
    Compute RT variability features for each repetition segment.

    Parameters:
    -----------
    imu_data : np.ndarray
        Full IMU data shaped (3, n_samples)
    repetition_segments : list
        List of (start_idx, end_idx) tuples defining repetition boundaries
    sampling_rate : float
        Sampling rate in Hz
    sensor_type : str
        Type of sensor ('acc' or 'gyr')
    site_name : str, optional
        Sensor site name for feature naming

    Returns:
    --------
    Dict[int, Dict[str, float]]
        Dictionary mapping repetition number to RT variability features
    """
    repetition_features = {}

    for rep_idx, (start_idx, end_idx) in enumerate(repetition_segments):
        # Extract data for this repetition
        rep_data = imu_data[:, start_idx:end_idx]

        if rep_data.shape[1] > 1:  # Need at least 2 samples for differences
            # Compute RT variability features for this repetition
            rep_features = compute_rt_variability_features_3axis(
                rep_data, sampling_rate, sensor_type, site_name
            )

            # Add repetition metadata
            rep_features[f"rep_duration{site_name if site_name else ''}"] = (end_idx - start_idx) / sampling_rate
            rep_features[f"rep_start_time{site_name if site_name else ''}"] = start_idx / sampling_rate
            rep_features[f"rep_end_time{site_name if site_name else ''}"] = end_idx / sampling_rate
            rep_features[f"rep_sample_count{site_name if site_name else ''}"] = end_idx - start_idx

            repetition_features[rep_idx + 1] = rep_features
        else:
            # Not enough data for RT variability analysis
            repetition_features[rep_idx + 1] = {}

    return repetition_features

