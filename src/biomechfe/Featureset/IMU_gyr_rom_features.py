"""
Range of Motion (ROM) feature extraction from 3-axis gyroscope data.
Includes statistical, temporal, biomechanical, and frequency domain features
specifically designed for movement analysis and clinical assessment.
"""

import numpy as np
from scipy.stats import skew, kurtosis
from scipy.fft import rfft, rfftfreq
from scipy.signal import welch, savgol_filter
from numpy.polynomial.polynomial import polyfit
from typing import Dict, Optional, Tuple


def compute_drift_corrected_angle(angular_velocity: np.ndarray, dt: float) -> np.ndarray:
    """
    Calculate angle with linear drift correction.

    Parameters:
    -----------
    angular_velocity : np.ndarray
        Angular velocity signal
    dt : float
        Time step (1/sampling_rate)

    Returns:
    --------
    np.ndarray
        Drift-corrected angle signal
    """
    if len(angular_velocity) < 2:
        return np.array([0.0])

    # Simple integration
    raw_angle = np.cumsum(angular_velocity) * dt

    # Linear drift correction
    t = np.arange(len(angular_velocity)) * dt
    try:
        # Fit linear trend
        p = np.polyfit(t, raw_angle, 1)
        # Remove linear trend (keeping only the constant term)
        drift_corrected = raw_angle - (p[0] * t)
    except:
        # If polyfit fails, return raw angle
        drift_corrected = raw_angle

    return drift_corrected


def compute_frequency_bands(psd: np.ndarray, freqs: np.ndarray) -> Dict[str, float]:
    """
    Calculate energy in different frequency bands.

    Parameters:
    -----------
    psd : np.ndarray
        Power spectral density
    freqs : np.ndarray
        Frequency array corresponding to PSD

    Returns:
    --------
    Dict[str, float]
        Dictionary of frequency band energies
    """
    bands = {
        'very_low': (0, 1),  # 0-1 Hz
        'low': (1, 3),  # 1-3 Hz
        'medium': (3, 7),  # 3-7 Hz
        'high': (7, 15)  # 7-15 Hz
    }

    band_energy = {}
    total_energy = np.sum(psd)

    if total_energy == 0:
        for band_name in bands.keys():
            band_energy[band_name] = 0.0
        return band_energy

    for band_name, (low, high) in bands.items():
        mask = (freqs >= low) & (freqs < high)
        if np.any(mask):
            band_energy[band_name] = np.sum(psd[mask]) / total_energy
        else:
            band_energy[band_name] = 0.0

    return band_energy


def compute_additional_biomechanical_features(
        arr: np.ndarray,
        sampling_rate: float
) -> Dict[str, float]:
    """
    Computes advanced biomechanical and frequency domain features for angular velocity.

    Parameters:
    -----------
    arr : np.ndarray
        1D array of angular velocities
    sampling_rate : float
        Sampling frequency in Hz

    Returns:
    --------
    Dict[str, float]
        Dictionary of biomechanical features
    """
    features = {}
    n = len(arr)

    if n < 1:
        # Handle edge case of empty array
        feature_names = [
            "Peak_AngVel", "TimeToPeak_AngVel", "CumulativeDisplacement",
            "AngVelRange", "SymmetryIndex", "SmoothnessIndex", "PSD_Energy",
            "DominantFreq", "MaxAngle", "MinAngle", "RangeAngle",
            "MeanAngle", "AngVel_RMS", "FatigueIndex", "MovementConsistency"
        ]
        for feature_name in feature_names:
            features[feature_name] = np.nan
        return features

    dt = 1.0 / sampling_rate

    # 1) Biomechanical Features

    # A) Peak Angular Velocity
    peak_val = np.max(arr)
    features["Peak_AngVel"] = peak_val

    # B) Time to Peak Angular Velocity
    idx_peak = np.argmax(arr)
    time_to_peak = idx_peak * dt
    features["TimeToPeak_AngVel"] = time_to_peak

    # C) Cumulative Angular Displacement = ∫ angular velocity dt
    cumulative_disp = np.sum(arr) * dt
    features["CumulativeDisplacement"] = cumulative_disp

    # D) Angular Velocity Range (RoM) = max(arr) - min(arr)
    angvel_range = np.max(arr) - np.min(arr)
    features["AngVelRange"] = angvel_range

    # E) Symmetry Index
    pos_part = arr[arr > 0].sum() * dt
    neg_part = arr[arr < 0].sum() * dt  # this is negative
    si_denom = np.abs(pos_part) + np.abs(neg_part)
    if si_denom == 0:
        symmetry_index = 0.0
    else:
        symmetry_index = (np.abs(pos_part) - np.abs(neg_part)) / si_denom
    features["SymmetryIndex"] = symmetry_index

    # F) Smoothness Index
    cutoff_freq = 5.0  # Hz
    try:
        freqs_full, psd_full = welch(arr, fs=sampling_rate, nperseg=min(256, n))
        # split low vs high
        low_mask = freqs_full <= cutoff_freq
        high_mask = freqs_full > cutoff_freq
        power_low = np.sum(psd_full[low_mask])
        power_high = np.sum(psd_full[high_mask]) if np.sum(high_mask) > 0 else 1e-9
        smoothness_index = power_low / (power_high + 1e-9)  # avoid div zero
        features["SmoothnessIndex"] = smoothness_index

        # 2) Frequency Domain Features
        total_energy = np.sum(psd_full)
        features["PSD_Energy"] = total_energy

        # Dominant Frequency: freq bin with highest PSD
        idx_dom = np.argmax(psd_full)
        dominant_freq = freqs_full[idx_dom]
        features["DominantFreq"] = dominant_freq

        # Frequency band energies
        band_energies = compute_frequency_bands(psd_full, freqs_full)
        for band, energy in band_energies.items():
            features[f"{band.capitalize()}FreqEnergy"] = energy

    except Exception:
        # If frequency analysis fails
        features["SmoothnessIndex"] = np.nan
        features["PSD_Energy"] = np.nan
        features["DominantFreq"] = np.nan
        for band in ['very_low', 'low', 'medium', 'high']:
            features[f"{band.capitalize()}FreqEnergy"] = np.nan

    # 3) Angle-Related Features (integrating velocity -> angle)
    try:
        angle_arr = compute_drift_corrected_angle(arr, dt)
        max_angle = np.max(angle_arr)
        min_angle = np.min(angle_arr)
        features["MaxAngle"] = max_angle
        features["MinAngle"] = min_angle
        features["RangeAngle"] = max_angle - min_angle
        features["MeanAngle"] = np.mean(angle_arr)
    except Exception:
        features["MaxAngle"] = np.nan
        features["MinAngle"] = np.nan
        features["RangeAngle"] = np.nan
        features["MeanAngle"] = np.nan

    # 4) Angular Velocity RMS
    angvel_rms = np.sqrt(np.mean(arr ** 2))
    features["AngVel_RMS"] = angvel_rms

    # 5) Fatigue Index: ratio of energy in first half vs second half of movement
    half_idx = len(arr) // 2
    if half_idx > 0:
        first_half_energy = np.sum(arr[:half_idx] ** 2)
        second_half_energy = np.sum(arr[half_idx:] ** 2)
        fatigue_index = first_half_energy / (second_half_energy + 1e-9)  # avoid div by zero
        features["FatigueIndex"] = fatigue_index
    else:
        features["FatigueIndex"] = np.nan

    # 6) Movement Consistency: correlation between first and second half
    if len(arr) >= 4:  # Need at least 2 points in each half
        first_half = arr[:half_idx]
        second_half = arr[half_idx:2 * half_idx]  # Use same length as first half
        if len(second_half) == len(first_half) and len(first_half) > 1:
            try:
                consistency = np.corrcoef(first_half, second_half)[0, 1]
                features["MovementConsistency"] = consistency
            except:
                features["MovementConsistency"] = np.nan
        else:
            features["MovementConsistency"] = np.nan
    else:
        features["MovementConsistency"] = np.nan

    return features


def compute_statistical_features(arr: np.ndarray) -> Dict[str, float]:
    """
    Compute basic statistical features for a signal.

    Parameters:
    -----------
    arr : np.ndarray
        Input signal array

    Returns:
    --------
    Dict[str, float]
        Dictionary of statistical features
    """
    features = {}

    if len(arr) == 0:
        feature_names = ['mean', 'median', 'min', 'max', 'range', 'std', 'var', 'skew', 'kurtosis']
        for metric in feature_names:
            features[metric] = np.nan
        return features

    arr_min = np.min(arr)
    arr_max = np.max(arr)
    arr_range = arr_max - arr_min

    features['mean'] = np.mean(arr)
    features['median'] = np.median(arr)
    features['min'] = arr_min
    features['max'] = arr_max
    features['range'] = arr_range
    features['std'] = np.std(arr, ddof=1) if len(arr) > 1 else 0.0
    features['var'] = np.var(arr, ddof=1) if len(arr) > 1 else 0.0
    features['skew'] = skew(arr) if len(arr) > 1 else 0.0
    features['kurtosis'] = kurtosis(arr) if len(arr) > 1 else 0.0

    return features


def compute_temporal_features(arr: np.ndarray, sampling_rate: float) -> Dict[str, float]:
    """
    Compute temporal features including slope, peak frequency, zero crossings, and extreme duration.

    Parameters:
    -----------
    arr : np.ndarray
        Input signal array
    sampling_rate : float
        Sampling rate in Hz

    Returns:
    --------
    Dict[str, float]
        Dictionary of temporal features
    """
    features = {}

    # A) Rate of Change (Slope of angular velocity)
    if len(arr) < 2:
        features['slope'] = np.nan
    else:
        t = np.arange(len(arr)) / sampling_rate
        try:
            # Fit a 1st-degree polynomial: arr ~ b + m*t. m is slope
            b, m = polyfit(t, arr, 1)  # polyfit returns [b, m] for a 1D poly (y = b + m*x)
            features['slope'] = m
        except:
            features['slope'] = np.nan

    # B) Peak Frequency (FFT)
    if len(arr) < 2:
        features['peak_freq'] = np.nan
    else:
        try:
            # rfft for real-valued signals
            fft_values = np.abs(rfft(arr))
            freqs = rfftfreq(len(arr), d=1.0 / sampling_rate)

            # Find index of maximum spectral amplitude (excluding the DC component at index 0)
            if len(fft_values) > 1:
                idx_peak = np.argmax(fft_values[1:]) + 1
                peak_freq = freqs[idx_peak]
                features['peak_freq'] = peak_freq
            else:
                features['peak_freq'] = 0.0
        except:
            features['peak_freq'] = np.nan

    # C) Number of Zero-Crossings
    if len(arr) < 2:
        features['zero_crossings'] = np.nan
    else:
        # Sign of each element
        signs = np.sign(arr)
        # Zero crossing if sign changes between consecutive samples
        zero_crosses = np.sum(np.diff(signs) != 0)
        features['zero_crossings'] = zero_crosses

    # D) Duration of Extreme Angles (or velocities)
    if len(arr) < 1:
        features['extreme_duration'] = np.nan
    else:
        # Use absolute velocities to capture both high + or - values
        abs_arr = np.abs(arr)
        # find threshold (90th percentile)
        threshold = np.percentile(abs_arr, 90)
        # count how many samples exceed threshold
        extreme_samples = np.sum(abs_arr >= threshold)
        # total time = extreme_samples / sampling_rate
        extreme_duration = extreme_samples / sampling_rate
        features['extreme_duration'] = extreme_duration

    return features


def compute_gyr_rom_features(
        gyr_data: np.ndarray,
        sampling_rate: float,
        site_name: Optional[str] = None
) -> Dict[str, float]:
    """
    Compute comprehensive ROM features for 3-axis gyroscope data.

    Parameters:
    -----------
    gyr_data : np.ndarray
        Gyroscope data shaped (3, n_samples) where rows are [X, Y, Z]
    sampling_rate : float
        Sampling rate in Hz
    site_name : str, optional
        Name of the sensor site for feature naming

    Returns:
    --------
    Dict[str, float]
        Dictionary of ROM features
    """
    if gyr_data.shape[0] != 3:
        raise ValueError(f"Expected gyr_data shape (3, n_samples), got {gyr_data.shape}")

    features = {}
    site_suffix = f"_{site_name}" if site_name else ""

    # Extract individual axes
    gx = gyr_data[0, :]
    gy = gyr_data[1, :]
    gz = gyr_data[2, :]

    # Compute magnitude
    g_mag = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)

    # Process each component (X, Y, Z, magnitude)
    components = {
        'X': gx,
        'Y': gy,
        'Z': gz,
        'mag': g_mag
    }

    for comp_name, data in components.items():
        prefix = f"ROM_{comp_name}{site_suffix}"

        # 1. Statistical Features
        stats = compute_statistical_features(data)
        for stat_name, stat_value in stats.items():
            features[f"{prefix}_{stat_name}"] = stat_value

        # 2. Temporal Features
        temporal = compute_temporal_features(data, sampling_rate)
        for temp_name, temp_value in temporal.items():
            features[f"{prefix}_{temp_name}"] = temp_value

        # 3. Additional biomechanical and frequency domain features
        biomech = compute_additional_biomechanical_features(data, sampling_rate)
        for biomech_name, biomech_value in biomech.items():
            features[f"{prefix}_{biomech_name}"] = biomech_value

    return features


def compute_gyr_rom_features_for_window(
        window: Dict,
        fs_imu: float,
        site_name: Optional[str] = None
) -> Dict[str, float]:
    """
    Convenience function to extract gyroscope ROM features from a window dict.

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
        Dictionary of gyroscope ROM features
    """
    if "gyr" not in window or window["gyr"] is None:
        return {}

    gyr_data = np.asarray(window["gyr"])

    return compute_gyr_rom_features(gyr_data, fs_imu, site_name)


def compute_gyr_rom_features_per_repetition(
        gyr_data: np.ndarray,
        repetition_segments: list,
        sampling_rate: float,
        site_name: Optional[str] = None
) -> Dict[int, Dict[str, float]]:
    """
    Compute ROM features for each repetition segment.

    Parameters:
    -----------
    gyr_data : np.ndarray
        Full gyroscope data shaped (3, n_samples)
    repetition_segments : list
        List of (start_idx, end_idx) tuples defining repetition boundaries
    sampling_rate : float
        Sampling rate in Hz
    site_name : str, optional
        Sensor site name for feature naming

    Returns:
    --------
    Dict[int, Dict[str, float]]
        Dictionary mapping repetition number to features
    """
    repetition_features = {}

    for rep_idx, (start_idx, end_idx) in enumerate(repetition_segments):
        # Extract data for this repetition
        rep_data = gyr_data[:, start_idx:end_idx]

        if rep_data.shape[1] > 0:  # Ensure we have data
            # Compute features for this repetition
            rep_features = compute_gyr_rom_features(rep_data, sampling_rate, site_name)

            # Add repetition metadata
            rep_features[f"rep_duration{site_name if site_name else ''}"] = (end_idx - start_idx) / sampling_rate
            rep_features[f"rep_start_time{site_name if site_name else ''}"] = start_idx / sampling_rate
            rep_features[f"rep_end_time{site_name if site_name else ''}"] = end_idx / sampling_rate

            repetition_features[rep_idx + 1] = rep_features
        else:
            # Empty repetition
            repetition_features[rep_idx + 1] = {}

    return repetition_features

