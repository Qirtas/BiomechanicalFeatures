"""
Integrated EMG (IEMG) feature extraction for biomechanical analysis.
Includes basic statistical features, fatigue-specific indicators, segmented analysis,
efficiency metrics, and frequency domain characteristics for muscle activation assessment.
"""

import numpy as np
from scipy.stats import skew, kurtosis
from typing import Dict, Optional

try:
    from scipy import signal

    SCIPY_SIGNAL_AVAILABLE = True
except ImportError:
    SCIPY_SIGNAL_AVAILABLE = False


def compute_emg_integrated_features(
        emg_signal: np.ndarray,
        sampling_rate: float,
        muscle_name: Optional[str] = None
) -> Dict[str, float]:
    """
    Compute comprehensive Integrated EMG (IEMG) features for a single EMG signal.

    Parameters:
    -----------
    emg_signal : np.ndarray
        1D array of EMG values
    sampling_rate : float
        Sampling rate of the EMG data in Hz
    muscle_name : str, optional
        Name of the muscle (e.g., 'biceps', 'deltoid') for feature naming

    Returns:
    --------
    Dict[str, float]
        Dictionary of IEMG features
    """
    features = {}
    prefix = f"{muscle_name}_" if muscle_name else ""

    # Safety check
    if emg_signal.size == 0:
        # Return NaNs for all features if empty
        feature_names = [
            "IEMG_Total", "IEMG_Per_Second", "IEMG_MAV", "IEMG_Variance", "IEMG_RMS",
            "IEMG_Peak", "IEMG_Median", "IEMG_25th", "IEMG_75th", "IEMG_Slope",
            "IEMG_Cumulative", "IEMG_TimeToPeak", "IEMG_NormalizedMean",
            # Fatigue-specific IEMG features
            "IEMG_FatigueIndex", "IEMG_RateOfChangeEarly", "IEMG_RateOfChangeLate",
            "IEMG_RateOfChangeRatio", "IEMG_ForceEfficiency",
            # Segmented IEMG analysis
            "IEMG_Phase1", "IEMG_Phase2", "IEMG_Phase3", "IEMG_PhaseRatio12",
            "IEMG_PhaseRatio23", "IEMG_PhaseRatio13",
            # Efficiency metrics
            "IEMG_EfficiencyIndex", "IEMG_SustainabilityIndex",
            # Frequency analysis
            "IEMG_FrequencyDispersion", "IEMG_FrequencyStability"
        ]
        for feature_name in feature_names:
            features[f"{prefix}{feature_name}"] = np.nan
        return features

    # 1) Absolute EMG (rectification)
    abs_emg = np.abs(emg_signal)

    # Duration of signal (in seconds)
    n_samples = len(emg_signal)
    duration = n_samples / sampling_rate

    # === BASIC IEMG FEATURES ===

    # 1) Total IEMG = sum of absolute EMG
    iemg_total = np.sum(abs_emg)
    features[f"{prefix}IEMG_Total"] = float(iemg_total)

    # 2) IEMG per second (normalized by duration)
    iemg_per_sec = iemg_total / duration if duration > 0 else np.nan
    features[f"{prefix}IEMG_Per_Second"] = float(iemg_per_sec)

    # 3) Mean Absolute Value (MAV)
    mav = np.mean(abs_emg)
    features[f"{prefix}IEMG_MAV"] = float(mav)

    # 4) Variance of absolute EMG
    iemg_var = np.var(abs_emg, ddof=1) if abs_emg.size > 1 else 0.0
    features[f"{prefix}IEMG_Variance"] = float(iemg_var)

    # 5) RMS of original signal
    rms_val = np.sqrt(np.mean(emg_signal ** 2))
    features[f"{prefix}IEMG_RMS"] = float(rms_val)

    # 6) Peak IEMG (max of absolute EMG)
    peak_val = np.max(abs_emg)
    features[f"{prefix}IEMG_Peak"] = float(peak_val)

    # 7) Median and percentiles of absolute EMG
    emg_median = np.median(abs_emg)
    features[f"{prefix}IEMG_Median"] = float(emg_median)
    emg_25th = np.percentile(abs_emg, 25)
    emg_75th = np.percentile(abs_emg, 75)
    features[f"{prefix}IEMG_25th"] = float(emg_25th)
    features[f"{prefix}IEMG_75th"] = float(emg_75th)

    # 8) IEMG slope (temporal trend)
    cumsum_emg = np.cumsum(abs_emg)
    if n_samples > 1:
        t = np.arange(n_samples) / sampling_rate
        # Linear fit of cumulative sum vs time
        slope_val = np.polyfit(t, cumsum_emg, 1)[0]  # slope coefficient
    else:
        slope_val = np.nan
    features[f"{prefix}IEMG_Slope"] = float(slope_val)

    # 9) Cumulative IEMG (final value of cumulative sum)
    features[f"{prefix}IEMG_Cumulative"] = float(cumsum_emg[-1])

    # 10) Time-to-Peak IEMG
    idx_peak = np.argmax(abs_emg)
    time_to_peak = idx_peak / sampling_rate
    features[f"{prefix}IEMG_TimeToPeak"] = float(time_to_peak)

    # 11) Normalized IEMG mean [0..1] within this signal
    min_val = np.min(abs_emg)
    max_val = np.max(abs_emg)
    if max_val > min_val:
        norm_emg = (abs_emg - min_val) / (max_val - min_val)
        normalized_mean = np.mean(norm_emg)
    else:
        # Edge case: all values are equal
        normalized_mean = 0.0
    features[f"{prefix}IEMG_NormalizedMean"] = float(normalized_mean)

    # === FATIGUE-SPECIFIC IEMG FEATURES ===

    # 1) IEMG Fatigue Index (ratio of IEMG in first vs. last part)
    if n_samples >= 4:  # Need at least 4 samples to divide into halves
        half_point = n_samples // 2
        first_half = abs_emg[:half_point]
        last_half = abs_emg[half_point:]

        iemg_first_half = np.sum(first_half)
        iemg_last_half = np.sum(last_half)

        # Fatigue index: ratio of first half to last half
        # Values > 1 indicate higher activation in early phase (potential fatigue)
        # Values < 1 indicate higher activation in later phase (compensation)
        if iemg_last_half > 0:
            fatigue_index = iemg_first_half / iemg_last_half
        else:
            fatigue_index = np.nan

        features[f"{prefix}IEMG_FatigueIndex"] = float(fatigue_index)
    else:
        features[f"{prefix}IEMG_FatigueIndex"] = np.nan

    # 2) IEMG Rate of Change Analysis
    if n_samples >= 6:  # Need enough samples for meaningful analysis
        # Divide signal into three equal parts
        third_point = n_samples // 3
        two_thirds_point = 2 * (n_samples // 3)

        # Calculate cumulative sum for each third
        cumsum_early = cumsum_emg[:third_point]
        cumsum_late = cumsum_emg[two_thirds_point:]

        # Calculate time arrays for each third
        t = np.arange(n_samples) / sampling_rate
        t_early = t[:third_point]
        t_late = t[two_thirds_point:]

        # Calculate slopes (rate of change) for each third
        if len(t_early) > 1:
            slope_early = np.polyfit(t_early, cumsum_early, 1)[0]
        else:
            slope_early = np.nan

        if len(t_late) > 1:
            slope_late = np.polyfit(t_late, cumsum_late, 1)[0]
        else:
            slope_late = np.nan

        # Store rates of change
        features[f"{prefix}IEMG_RateOfChangeEarly"] = float(slope_early)
        features[f"{prefix}IEMG_RateOfChangeLate"] = float(slope_late)

        # Rate of change ratio (early/late) - indicator of fatigue development
        if slope_late > 0:
            roc_ratio = slope_early / slope_late
        else:
            roc_ratio = np.nan

        features[f"{prefix}IEMG_RateOfChangeRatio"] = float(roc_ratio)
    else:
        features[f"{prefix}IEMG_RateOfChangeEarly"] = np.nan
        features[f"{prefix}IEMG_RateOfChangeLate"] = np.nan
        features[f"{prefix}IEMG_RateOfChangeRatio"] = np.nan

    # 3) IEMG-to-Force Relationship Metrics
    # Note: Without direct force measurements, we use peak EMG as proxy for maximum force
    if peak_val > 0:
        # IEMG efficiency: total IEMG relative to peak activation
        # Lower values may indicate more efficient muscle activation
        iemg_force_efficiency = iemg_total / (peak_val * n_samples)
        features[f"{prefix}IEMG_ForceEfficiency"] = float(iemg_force_efficiency)
    else:
        features[f"{prefix}IEMG_ForceEfficiency"] = np.nan

    # === SEGMENTED IEMG ANALYSIS ===

    # Divide signal into three phases for temporal analysis
    if n_samples >= 6:  # Need enough samples for meaningful segmentation
        third_point = n_samples // 3
        two_thirds_point = 2 * (n_samples // 3)

        # Calculate IEMG for each phase
        phase1 = np.sum(abs_emg[:third_point])
        phase2 = np.sum(abs_emg[third_point:two_thirds_point])
        phase3 = np.sum(abs_emg[two_thirds_point:])

        # Store phase IEMGs
        features[f"{prefix}IEMG_Phase1"] = float(phase1)
        features[f"{prefix}IEMG_Phase2"] = float(phase2)
        features[f"{prefix}IEMG_Phase3"] = float(phase3)

        # Calculate phase ratios (important for detecting fatigue patterns)
        if phase2 > 0:
            phase_ratio12 = phase1 / phase2
        else:
            phase_ratio12 = np.nan

        if phase3 > 0:
            phase_ratio23 = phase2 / phase3
            phase_ratio13 = phase1 / phase3
        else:
            phase_ratio23 = np.nan
            phase_ratio13 = np.nan

        features[f"{prefix}IEMG_PhaseRatio12"] = float(phase_ratio12)
        features[f"{prefix}IEMG_PhaseRatio23"] = float(phase_ratio23)
        features[f"{prefix}IEMG_PhaseRatio13"] = float(phase_ratio13)
    else:
        features[f"{prefix}IEMG_Phase1"] = np.nan
        features[f"{prefix}IEMG_Phase2"] = np.nan
        features[f"{prefix}IEMG_Phase3"] = np.nan
        features[f"{prefix}IEMG_PhaseRatio12"] = np.nan
        features[f"{prefix}IEMG_PhaseRatio23"] = np.nan
        features[f"{prefix}IEMG_PhaseRatio13"] = np.nan

    # === IEMG-BASED MUSCLE EFFICIENCY METRICS ===

    # 1) Efficiency Index: Relationship between IEMG and its variability
    # Lower variability for same IEMG may indicate more efficient muscle recruitment
    if iemg_total > 0:
        efficiency_index = iemg_var / iemg_total
        features[f"{prefix}IEMG_EfficiencyIndex"] = float(efficiency_index)
    else:
        features[f"{prefix}IEMG_EfficiencyIndex"] = np.nan

    # 2) Sustainability Index: Ability to maintain activation level
    # Compares early vs late activation patterns
    if n_samples >= 4:
        # Calculate mean activation in first and last quarter
        quarter_point = n_samples // 4
        first_quarter_mean = np.mean(abs_emg[:quarter_point])
        last_quarter_mean = np.mean(abs_emg[-quarter_point:])

        if first_quarter_mean > 0:
            sustainability_index = last_quarter_mean / first_quarter_mean
        else:
            sustainability_index = np.nan

        features[f"{prefix}IEMG_SustainabilityIndex"] = float(sustainability_index)
    else:
        features[f"{prefix}IEMG_SustainabilityIndex"] = np.nan

    # === IEMG FREQUENCY ANALYSIS ===

    # Analyze the rate of change in IEMG for frequency characteristics
    if n_samples >= 10 and SCIPY_SIGNAL_AVAILABLE:  # Need enough samples for frequency analysis
        # Calculate the derivative of IEMG (rate of change)
        iemg_derivative = np.diff(cumsum_emg)

        # Ensure we have enough samples for frequency analysis
        if len(iemg_derivative) >= 8:
            try:
                # Use Welch's method to estimate power spectral density
                f, psd = signal.welch(iemg_derivative, fs=sampling_rate,
                                      nperseg=min(256, len(iemg_derivative)))

                # Calculate frequency dispersion (spectral spread)
                if np.sum(psd) > 0:
                    # Normalized PSD
                    norm_psd = psd / np.sum(psd)

                    # Calculate centroid frequency
                    centroid_freq = np.sum(f * norm_psd)

                    # Calculate frequency dispersion (spectral spread)
                    freq_dispersion = np.sqrt(np.sum(((f - centroid_freq) ** 2) * norm_psd))

                    # Calculate frequency stability (ratio of power in low vs high frequencies)
                    # Higher values indicate more stable, less erratic activation patterns
                    low_freq_idx = f <= 5  # Low frequency components (< 5 Hz)
                    high_freq_idx = f > 5  # High frequency components (> 5 Hz)

                    low_freq_power = np.sum(psd[low_freq_idx]) if np.any(low_freq_idx) else 0
                    high_freq_power = np.sum(psd[high_freq_idx]) if np.any(high_freq_idx) else 0

                    if high_freq_power > 0:
                        freq_stability = low_freq_power / high_freq_power
                    else:
                        freq_stability = np.nan

                    features[f"{prefix}IEMG_FrequencyDispersion"] = float(freq_dispersion)
                    features[f"{prefix}IEMG_FrequencyStability"] = float(freq_stability)
                else:
                    features[f"{prefix}IEMG_FrequencyDispersion"] = np.nan
                    features[f"{prefix}IEMG_FrequencyStability"] = np.nan
            except Exception:
                features[f"{prefix}IEMG_FrequencyDispersion"] = np.nan
                features[f"{prefix}IEMG_FrequencyStability"] = np.nan
        else:
            features[f"{prefix}IEMG_FrequencyDispersion"] = np.nan
            features[f"{prefix}IEMG_FrequencyStability"] = np.nan
    else:
        features[f"{prefix}IEMG_FrequencyDispersion"] = np.nan
        features[f"{prefix}IEMG_FrequencyStability"] = np.nan

    return features


def compute_emg_integrated_features_multi_channel(
        emg_data: np.ndarray,
        sampling_rate: float,
        muscle_names: Optional[list] = None
) -> Dict[str, float]:
    """
    Compute Integrated EMG features for multi-channel EMG data.

    Parameters:
    -----------
    emg_data : np.ndarray
        EMG data shaped (n_channels, n_samples)
    sampling_rate : float
        Sampling rate of the EMG data in Hz
    muscle_names : list, optional
        List of muscle names for each channel. If None, uses generic channel names.

    Returns:
    --------
    Dict[str, float]
        Dictionary of IEMG features for all channels
    """
    if emg_data.ndim != 2:
        raise ValueError(f"Expected emg_data shape (n_channels, n_samples), got {emg_data.shape}")

    features = {}
    n_channels = emg_data.shape[0]

    # Set muscle names
    if muscle_names is None:
        muscle_names = [f"ch{i}" for i in range(n_channels)]
    elif len(muscle_names) != n_channels:
        raise ValueError(f"Number of muscle names ({len(muscle_names)}) must match number of channels ({n_channels})")

    # Process each channel
    for ch_idx, muscle_name in enumerate(muscle_names):
        channel_data = emg_data[ch_idx, :]

        if channel_data.size == 0:
            continue

        # Compute IEMG features for this channel
        channel_features = compute_emg_integrated_features(
            channel_data, sampling_rate, muscle_name
        )
        features.update(channel_features)

    # Compute features for the mean across all channels (if multiple channels)
    if n_channels > 1:
        mean_emg = np.mean(emg_data, axis=0)
        mean_features = compute_emg_integrated_features(
            mean_emg, sampling_rate, "mean"
        )
        features.update(mean_features)

    return features


def compute_emg_integrated_features_for_window(
        window: Dict,
        fs_emg: float,
        muscle_names: Optional[list] = None
) -> Dict[str, float]:
    """
    Convenience function to extract Integrated EMG features from a window dict.

    Parameters:
    -----------
    window : Dict
        Window dictionary containing 'emg' key with shape (n_channels, n_samples)
    fs_emg : float
        EMG sampling frequency
    muscle_names : list, optional
        List of muscle names for each channel

    Returns:
    --------
    Dict[str, float]
        Dictionary of Integrated EMG features
    """
    if "emg" not in window or window["emg"] is None:
        return {}

    emg_data = np.asarray(window["emg"])

    # Handle single channel case
    if emg_data.ndim == 1:
        emg_data = emg_data[np.newaxis, :]  # Add channel dimension

    return compute_emg_integrated_features_multi_channel(
        emg_data, fs_emg, muscle_names
    )


def compute_emg_integrated_features_per_repetition(
        emg_data: np.ndarray,
        repetition_segments: list,
        sampling_rate: float,
        muscle_names: Optional[list] = None
) -> Dict[int, Dict[str, float]]:
    """
    Compute Integrated EMG features for each repetition segment.

    Parameters:
    -----------
    emg_data : np.ndarray
        Full EMG data shaped (n_channels, n_samples)
    repetition_segments : list
        List of (start_idx, end_idx) tuples defining repetition boundaries
    sampling_rate : float
        Sampling rate in Hz
    muscle_names : list, optional
        List of muscle names for each channel

    Returns:
    --------
    Dict[int, Dict[str, float]]
        Dictionary mapping repetition number to IEMG features
    """
    repetition_features = {}

    for rep_idx, (start_idx, end_idx) in enumerate(repetition_segments):
        # Extract data for this repetition
        rep_data = emg_data[:, start_idx:end_idx]

        if rep_data.shape[1] > 0:  # Ensure we have data
            # Compute IEMG features for this repetition
            rep_features = compute_emg_integrated_features_multi_channel(
                rep_data, sampling_rate, muscle_names
            )

            # Add repetition metadata
            rep_features["rep_duration"] = (end_idx - start_idx) / sampling_rate
            rep_features["rep_start_time"] = start_idx / sampling_rate
            rep_features["rep_end_time"] = end_idx / sampling_rate
            rep_features["rep_sample_count"] = end_idx - start_idx

            repetition_features[rep_idx + 1] = rep_features
        else:
            # Empty repetition
            repetition_features[rep_idx + 1] = {}

    return repetition_features
