"""
Wavelength (WV) feature extraction for EMG signals.
Includes waveform length analysis, fatigue-specific indicators, burst analysis,
normalized features, and advanced wavelength-based characteristics for muscle activation assessment.
"""

import numpy as np
from scipy.stats import skew, kurtosis
from typing import Dict, Optional

try:
    from scipy.signal import savgol_filter

    SCIPY_SIGNAL_AVAILABLE = True
except ImportError:
    SCIPY_SIGNAL_AVAILABLE = False


def compute_emg_wavelength_features_for_signal(
        emg_signal: np.ndarray,
        sampling_rate: float = 1000,
        muscle_name: Optional[str] = None
) -> Dict[str, float]:
    """
    Compute comprehensive wavelength features for a single EMG signal.

    Parameters:
    -----------
    emg_signal : np.ndarray
        1D array of EMG values
    sampling_rate : float, default 1000
        Sampling rate of the EMG data in Hz
    muscle_name : str, optional
        Name of the muscle for feature naming

    Returns:
    --------
    Dict[str, float]
        Dictionary of wavelength features
    """
    features = {}
    prefix = f"{muscle_name}_" if muscle_name else ""

    # Safety check
    if emg_signal.size == 0 or np.all(np.isnan(emg_signal)):
        placeholders = [
            "WV_Total", "WV_Per_Second", "WV_Slope", "TimeToHalfWV", "WV_Variance",
            "WV_Segment1", "WV_Segment2", "WV_Segment3", "WV_Ratio12", "WV_Ratio23", "WV_Ratio13",
            "WV_NormByRange", "WV_NormByStd", "WV_ComplexityIndex", "WV_Acceleration",
            "WV_FatigueRatio", "WV_BurstCount", "WV_MeanBurstDuration", "WV_MeanBurstAmplitude"
        ]
        for pl in placeholders:
            features[f"{prefix}{pl}"] = np.nan
        return features

    # Clean input data
    emg_signal = np.nan_to_num(emg_signal, nan=0.0)

    # Duration of the signal (in seconds)
    n_samples = len(emg_signal)
    duration = n_samples / sampling_rate

    # 1) Compute Waveform Length (WV_Total)
    #    WV = sum of absolute differences between consecutive samples
    if n_samples < 2:
        # Need at least 2 samples to compute differences
        for pl in [
            "WV_Total", "WV_Per_Second", "WV_Slope", "TimeToHalfWV", "WV_Variance",
            "WV_Segment1", "WV_Segment2", "WV_Segment3", "WV_Ratio12", "WV_Ratio23", "WV_Ratio13",
            "WV_NormByRange", "WV_NormByStd", "WV_ComplexityIndex", "WV_Acceleration",
            "WV_FatigueRatio", "WV_BurstCount", "WV_MeanBurstDuration", "WV_MeanBurstAmplitude"
        ]:
            features[f"{prefix}{pl}"] = np.nan
        return features

    abs_diff = np.abs(np.diff(emg_signal))
    wv_total = np.sum(abs_diff)
    features[f"{prefix}WV_Total"] = float(wv_total)

    # 2) WV per second (WV_Total / signal duration)
    wv_per_second = wv_total / duration if duration > 0 else np.nan
    features[f"{prefix}WV_Per_Second"] = float(wv_per_second)

    # 3) WV_Slope
    #    Consider the cumulative sum of abs_diff over time and fit a line
    #    to see how quickly the wave length accumulates.
    cumsum_wv = np.cumsum(abs_diff)
    if len(abs_diff) > 1:
        # Create a time vector with n_samples - 1 points (since we used diff)
        t = np.arange(len(abs_diff)) / sampling_rate
        try:
            slope = np.polyfit(t, cumsum_wv, 1)[0]  # Linear fit slope
        except:
            slope = np.nan
    else:
        slope = np.nan
    features[f"{prefix}WV_Slope"] = float(slope)

    # 4) TimeToHalfWV
    #    The time at which the cumulative wave length is 50% of wv_total
    if wv_total > 0:
        half_wv = wv_total / 2.0
        idx_half = np.searchsorted(cumsum_wv, half_wv)
        if idx_half < len(cumsum_wv):
            time_to_half_wv = idx_half / sampling_rate
        else:
            time_to_half_wv = duration
    else:
        time_to_half_wv = np.nan
    features[f"{prefix}TimeToHalfWV"] = float(time_to_half_wv)

    # 5) WV_Variance
    #    Variance in abs_diff can serve as a measure of EMG complexity changes.
    if len(abs_diff) > 1:
        wv_variance = np.var(abs_diff, ddof=1)  # sample variance
    else:
        wv_variance = 0.0
    features[f"{prefix}WV_Variance"] = float(wv_variance)

    # 6) Segmented Wavelength Analysis
    # Divide the signal into three equal parts to track changes within movement
    if n_samples >= 6:  # Need enough samples for meaningful segmentation
        third_point = n_samples // 3
        two_thirds_point = 2 * (n_samples // 3)

        # Calculate wavelength for each segment
        if third_point > 0:
            segment1 = emg_signal[:third_point]
            if len(segment1) > 1:
                abs_diff1 = np.abs(np.diff(segment1))
                wv_segment1 = np.sum(abs_diff1)
            else:
                wv_segment1 = np.nan
            features[f"{prefix}WV_Segment1"] = float(wv_segment1)
        else:
            features[f"{prefix}WV_Segment1"] = np.nan

        if third_point < two_thirds_point:
            segment2 = emg_signal[third_point:two_thirds_point]
            if len(segment2) > 1:
                abs_diff2 = np.abs(np.diff(segment2))
                wv_segment2 = np.sum(abs_diff2)
            else:
                wv_segment2 = np.nan
            features[f"{prefix}WV_Segment2"] = float(wv_segment2)
        else:
            features[f"{prefix}WV_Segment2"] = np.nan

        if two_thirds_point < n_samples:
            segment3 = emg_signal[two_thirds_point:]
            if len(segment3) > 1:
                abs_diff3 = np.abs(np.diff(segment3))
                wv_segment3 = np.sum(abs_diff3)
            else:
                wv_segment3 = np.nan
            features[f"{prefix}WV_Segment3"] = float(wv_segment3)
        else:
            features[f"{prefix}WV_Segment3"] = np.nan

        # Calculate segment ratios (important for detecting fatigue patterns)
        try:
            if not np.isnan(wv_segment1) and not np.isnan(wv_segment2) and wv_segment2 > 0:
                wv_ratio12 = wv_segment1 / wv_segment2
            else:
                wv_ratio12 = np.nan

            if not np.isnan(wv_segment2) and not np.isnan(wv_segment3) and wv_segment3 > 0:
                wv_ratio23 = wv_segment2 / wv_segment3
            else:
                wv_ratio23 = np.nan

            if not np.isnan(wv_segment1) and not np.isnan(wv_segment3) and wv_segment3 > 0:
                wv_ratio13 = wv_segment1 / wv_segment3
            else:
                wv_ratio13 = np.nan
        except:
            wv_ratio12 = np.nan
            wv_ratio23 = np.nan
            wv_ratio13 = np.nan

        features[f"{prefix}WV_Ratio12"] = float(wv_ratio12)
        features[f"{prefix}WV_Ratio23"] = float(wv_ratio23)
        features[f"{prefix}WV_Ratio13"] = float(wv_ratio13)
    else:
        features[f"{prefix}WV_Segment1"] = np.nan
        features[f"{prefix}WV_Segment2"] = np.nan
        features[f"{prefix}WV_Segment3"] = np.nan
        features[f"{prefix}WV_Ratio12"] = np.nan
        features[f"{prefix}WV_Ratio23"] = np.nan
        features[f"{prefix}WV_Ratio13"] = np.nan

    # 7) Normalized Wavelength Features
    # Normalize wavelength by signal amplitude to account for amplitude variations
    if n_samples > 0:
        # Calculate signal amplitude metrics
        signal_range = np.max(emg_signal) - np.min(emg_signal)
        signal_std = np.std(emg_signal)

        # Normalize wavelength by range and standard deviation
        if signal_range > 0:
            wv_norm_range = wv_total / signal_range
        else:
            wv_norm_range = np.nan

        if signal_std > 0:
            wv_norm_std = wv_total / signal_std
        else:
            wv_norm_std = np.nan

        features[f"{prefix}WV_NormByRange"] = float(wv_norm_range)
        features[f"{prefix}WV_NormByStd"] = float(wv_norm_std)
    else:
        features[f"{prefix}WV_NormByRange"] = np.nan
        features[f"{prefix}WV_NormByStd"] = np.nan

    # 8) Wavelength Fatigue Indices
    # Calculate specific fatigue-related indices based on wavelength

    # Wavelength Complexity Index: ratio of variance to mean of abs_diff
    # Higher values indicate more irregular patterns (often seen with fatigue)
    if len(abs_diff) > 0:
        abs_diff_mean = np.mean(abs_diff)
        if abs_diff_mean > 0:
            wv_complexity_index = wv_variance / abs_diff_mean
        else:
            wv_complexity_index = np.nan
    else:
        wv_complexity_index = np.nan

    features[f"{prefix}WV_ComplexityIndex"] = float(wv_complexity_index)

    # Wavelength Acceleration: second derivative of cumulative wavelength
    # Indicates changes in the rate of complexity accumulation
    if len(abs_diff) >= 3:
        # First derivative is abs_diff, second derivative is diff of abs_diff
        wv_acceleration = np.mean(np.abs(np.diff(abs_diff)))
    else:
        wv_acceleration = np.nan

    features[f"{prefix}WV_Acceleration"] = float(wv_acceleration)

    # Wavelength Fatigue Ratio: ratio of early to late wavelength
    # Decreases with fatigue as later portions become more complex
    if n_samples >= 4:
        half_point = n_samples // 2
        early_segment = emg_signal[:half_point]
        late_segment = emg_signal[half_point:]

        if len(early_segment) > 1 and len(late_segment) > 1:
            early_wv = np.sum(np.abs(np.diff(early_segment)))
            late_wv = np.sum(np.abs(np.diff(late_segment)))

            if late_wv > 0:
                wv_fatigue_ratio = early_wv / late_wv
            else:
                wv_fatigue_ratio = np.nan
        else:
            wv_fatigue_ratio = np.nan
    else:
        wv_fatigue_ratio = np.nan

    features[f"{prefix}WV_FatigueRatio"] = float(wv_fatigue_ratio)

    # 9) Wavelength Burst Analysis
    # Detect bursts of high wavelength activity (periods of rapid signal changes)
    if len(abs_diff) >= 10:
        try:
            # Define burst threshold as 2x the median wavelength
            median_diff = np.median(abs_diff)
            burst_threshold = 2 * median_diff

            # Find regions where wavelength exceeds threshold
            above_threshold = abs_diff > burst_threshold

            # Count bursts (consecutive regions above threshold)
            burst_count = 0
            burst_durations = []
            burst_amplitudes = []

            in_burst = False
            burst_start = 0

            for i, is_above in enumerate(above_threshold):
                if is_above and not in_burst:
                    # Start of a new burst
                    in_burst = True
                    burst_start = i
                elif not is_above and in_burst:
                    # End of a burst
                    in_burst = False
                    burst_count += 1
                    burst_duration = (i - burst_start) / sampling_rate
                    burst_durations.append(burst_duration)
                    if burst_start < i:
                        burst_amplitudes.append(np.mean(abs_diff[burst_start:i]))

            # Handle case where signal ends during a burst
            if in_burst:
                burst_count += 1
                burst_duration = (len(above_threshold) - burst_start) / sampling_rate
                burst_durations.append(burst_duration)
                if burst_start < len(abs_diff):
                    burst_amplitudes.append(np.mean(abs_diff[burst_start:]))

            features[f"{prefix}WV_BurstCount"] = float(burst_count)
            features[f"{prefix}WV_MeanBurstDuration"] = float(np.mean(burst_durations)) if burst_durations else np.nan
            features[f"{prefix}WV_MeanBurstAmplitude"] = float(
                np.mean(burst_amplitudes)) if burst_amplitudes else np.nan
        except Exception:
            features[f"{prefix}WV_BurstCount"] = np.nan
            features[f"{prefix}WV_MeanBurstDuration"] = np.nan
            features[f"{prefix}WV_MeanBurstAmplitude"] = np.nan
    else:
        features[f"{prefix}WV_BurstCount"] = np.nan
        features[f"{prefix}WV_MeanBurstDuration"] = np.nan
        features[f"{prefix}WV_MeanBurstAmplitude"] = np.nan

    return features


def compute_emg_wavelength_features_multi_channel(
        emg_data: np.ndarray,
        sampling_rate: float,
        muscle_names: Optional[list] = None
) -> Dict[str, float]:
    """
    Compute wavelength features for multi-channel EMG data.

    Parameters:
    -----------
    emg_data : np.ndarray
        EMG data shaped (n_channels, n_samples)
    sampling_rate : float
        Sampling rate of the EMG data in Hz
    muscle_names : list, optional
        List of muscle names for each channel

    Returns:
    --------
    Dict[str, float]
        Dictionary of wavelength features for all channels
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

        # Compute wavelength features for this channel
        channel_features = compute_emg_wavelength_features_for_signal(
            channel_data, sampling_rate, muscle_name
        )
        features.update(channel_features)

    # Compute features for the mean across all channels (if multiple channels)
    if n_channels > 1:
        mean_emg = np.mean(emg_data, axis=0)
        mean_features = compute_emg_wavelength_features_for_signal(
            mean_emg, sampling_rate, "mean"
        )
        features.update(mean_features)

    return features


def compute_emg_wavelength_features_for_window(
        window: Dict,
        fs_emg: float,
        muscle_names: Optional[list] = None
) -> Dict[str, float]:
    """
    Convenience function to extract wavelength EMG features from a window dict.

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
        Dictionary of wavelength EMG features
    """
    if "emg" not in window or window["emg"] is None:
        return {}

    emg_data = np.asarray(window["emg"])

    # Handle single channel case
    if emg_data.ndim == 1:
        emg_data = emg_data[np.newaxis, :]  # Add channel dimension

    return compute_emg_wavelength_features_multi_channel(
        emg_data, fs_emg, muscle_names
    )


def compute_emg_wavelength_features_per_repetition(
        emg_data: np.ndarray,
        repetition_segments: list,
        sampling_rate: float,
        muscle_names: Optional[list] = None
) -> Dict[int, Dict[str, float]]:
    """
    Compute wavelength EMG features for each repetition segment.

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
        Dictionary mapping repetition number to wavelength features
    """
    repetition_features = {}

    for rep_idx, (start_idx, end_idx) in enumerate(repetition_segments):
        # Extract data for this repetition
        rep_data = emg_data[:, start_idx:end_idx]

        if rep_data.shape[1] > 1:  # Need at least 2 samples for wavelength
            # Compute wavelength features for this repetition
            rep_features = compute_emg_wavelength_features_multi_channel(
                rep_data, sampling_rate, muscle_names
            )

            # Add repetition metadata
            rep_features["rep_duration"] = (end_idx - start_idx) / sampling_rate
            rep_features["rep_start_time"] = start_idx / sampling_rate
            rep_features["rep_end_time"] = end_idx / sampling_rate
            rep_features["rep_sample_count"] = end_idx - start_idx

            repetition_features[rep_idx + 1] = rep_features
        else:
            # Not enough data for wavelength analysis
            repetition_features[rep_idx + 1] = {}

    return repetition_features

