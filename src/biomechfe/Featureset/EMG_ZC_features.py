"""
Zero Crossing (ZC) feature extraction for EMG signals.
Includes standard and threshold-based zero crossings, segmented analysis,
frequency band analysis, fatigue indicators, and trend analysis for muscle activation assessment.
"""

import numpy as np
from scipy.stats import linregress
from typing import Dict, Optional

try:
    from scipy.signal import butter, filtfilt

    SCIPY_SIGNAL_AVAILABLE = True
except ImportError:
    SCIPY_SIGNAL_AVAILABLE = False


def compute_emg_zc_features_for_signal(
        emg_signal: np.ndarray,
        sampling_rate: float = 1000,
        muscle_name: Optional[str] = None,
        threshold: float = 0.01
) -> Dict[str, float]:
    """
    Compute comprehensive Zero Crossing (ZC) features for a single EMG signal.

    Parameters:
    -----------
    emg_signal : np.ndarray
        1D array of EMG values
    sampling_rate : float, default 1000
        Sampling rate of the EMG data in Hz
    muscle_name : str, optional
        Name of the muscle for feature naming
    threshold : float, default 0.01
        Threshold to reduce noise sensitivity

    Returns:
    --------
    Dict[str, float]
        Dictionary of zero crossing features
    """
    features = {}
    prefix = f"{muscle_name}_" if muscle_name else ""

    # Safety check
    if emg_signal.size < 2:
        placeholders = [
            "ZC_Total", "ZC_Rate", "ZC_MeanInterval", "ZC_StdInterval", "ZC_MinInterval", "ZC_MaxInterval",
            "ZC_ThreshTotal", "ZC_ThreshRate", "ZC_ThreshMeanInterval", "ZC_ThreshStdInterval",
            "ZC_ThreshMinInterval", "ZC_ThreshMaxInterval", "ZC_RateSegment1", "ZC_RateSegment2",
            "ZC_RateSegment3", "ZC_RateRatio12", "ZC_RateRatio23", "ZC_RateRatio13",
            "ZC_LowBand_Total", "ZC_LowBand_Rate", "ZC_MidBand_Total", "ZC_MidBand_Rate",
            "ZC_HighBand_Total", "ZC_HighBand_Rate", "ZC_LowToHighRatio", "ZC_FatigueIndex",
            "ZC_EarlyRate", "ZC_LateRate", "ZC_TrendSlope", "ZC_RateChange"
        ]
        for pl in placeholders:
            features[f"{prefix}{pl}"] = np.nan
        return features

    # Clean input data
    emg_signal = np.nan_to_num(emg_signal, nan=0.0)

    # Duration of the signal (in seconds)
    duration_sec = emg_signal.size / sampling_rate

    # 1) Standard Zero Crossing Features
    # Compute zero crossings: A zero crossing occurs when the sign of the signal changes.
    zc_indices = np.where(np.diff(np.sign(emg_signal)) != 0)[0]
    zc_total = zc_indices.size
    features[f"{prefix}ZC_Total"] = zc_total

    # Zero crossing rate (number per second)
    zc_rate = zc_total / duration_sec if duration_sec > 0 else np.nan
    features[f"{prefix}ZC_Rate"] = zc_rate

    # Compute intervals between zero crossings (in seconds)
    if zc_total > 1:
        intervals = np.diff(zc_indices) / sampling_rate
        features[f"{prefix}ZC_MeanInterval"] = np.mean(intervals)
        features[f"{prefix}ZC_StdInterval"] = np.std(intervals, ddof=1) if intervals.size > 1 else 0.0
        features[f"{prefix}ZC_MinInterval"] = np.min(intervals)
        features[f"{prefix}ZC_MaxInterval"] = np.max(intervals)
    else:
        # Not enough crossings to compute intervals.
        features[f"{prefix}ZC_MeanInterval"] = np.nan
        features[f"{prefix}ZC_StdInterval"] = np.nan
        features[f"{prefix}ZC_MinInterval"] = np.nan
        features[f"{prefix}ZC_MaxInterval"] = np.nan

    # 2) Threshold-Based Zero Crossing Features
    # Only count zero crossings if the signal crosses zero by more than the threshold
    # This reduces sensitivity to noise

    # Calculate signal amplitude for adaptive threshold
    signal_range = np.max(emg_signal) - np.min(emg_signal)
    adaptive_threshold = threshold * signal_range

    # Find consecutive samples that cross zero by more than the threshold
    thresh_zc_indices = []
    for i in range(len(emg_signal) - 1):
        if (emg_signal[i] * emg_signal[i + 1] < 0) and (
                abs(emg_signal[i]) > adaptive_threshold or abs(emg_signal[i + 1]) > adaptive_threshold):
            thresh_zc_indices.append(i)

    thresh_zc_indices = np.array(thresh_zc_indices)
    thresh_zc_total = len(thresh_zc_indices)

    features[f"{prefix}ZC_ThreshTotal"] = thresh_zc_total

    # Threshold-based zero crossing rate
    thresh_zc_rate = thresh_zc_total / duration_sec if duration_sec > 0 else np.nan
    features[f"{prefix}ZC_ThreshRate"] = thresh_zc_rate

    # Compute intervals between threshold-based zero crossings
    if thresh_zc_total > 1:
        thresh_intervals = np.diff(thresh_zc_indices) / sampling_rate
        features[f"{prefix}ZC_ThreshMeanInterval"] = np.mean(thresh_intervals)
        features[f"{prefix}ZC_ThreshStdInterval"] = np.std(thresh_intervals,
                                                           ddof=1) if thresh_intervals.size > 1 else 0.0
        features[f"{prefix}ZC_ThreshMinInterval"] = np.min(thresh_intervals)
        features[f"{prefix}ZC_ThreshMaxInterval"] = np.max(thresh_intervals)
    else:
        features[f"{prefix}ZC_ThreshMeanInterval"] = np.nan
        features[f"{prefix}ZC_ThreshStdInterval"] = np.nan
        features[f"{prefix}ZC_ThreshMinInterval"] = np.nan
        features[f"{prefix}ZC_ThreshMaxInterval"] = np.nan

    # 3) Segmented Zero Crossing Analysis
    # Divide the signal into three equal parts to track changes within repetition
    if emg_signal.size >= 6:  # Need enough samples for meaningful segmentation
        third_point = emg_signal.size // 3
        two_thirds_point = 2 * (emg_signal.size // 3)

        # Calculate zero crossings for each segment
        segment1 = emg_signal[:third_point]
        segment2 = emg_signal[third_point:two_thirds_point]
        segment3 = emg_signal[two_thirds_point:]

        # Zero crossings in each segment
        zc_segment1 = np.sum(np.diff(np.sign(segment1)) != 0)
        zc_segment2 = np.sum(np.diff(np.sign(segment2)) != 0)
        zc_segment3 = np.sum(np.diff(np.sign(segment3)) != 0)

        # Normalize by segment duration
        seg_duration = third_point / sampling_rate
        zc_rate_segment1 = zc_segment1 / seg_duration if seg_duration > 0 else np.nan
        zc_rate_segment2 = zc_segment2 / seg_duration if seg_duration > 0 else np.nan
        zc_rate_segment3 = zc_segment3 / seg_duration if seg_duration > 0 else np.nan

        features[f"{prefix}ZC_RateSegment1"] = zc_rate_segment1
        features[f"{prefix}ZC_RateSegment2"] = zc_rate_segment2
        features[f"{prefix}ZC_RateSegment3"] = zc_rate_segment3

        # Calculate segment ratios (important for detecting fatigue patterns)
        if zc_rate_segment2 > 0:
            zc_ratio12 = zc_rate_segment1 / zc_rate_segment2
        else:
            zc_ratio12 = np.nan

        if zc_rate_segment3 > 0:
            zc_ratio23 = zc_rate_segment2 / zc_rate_segment3
            zc_ratio13 = zc_rate_segment1 / zc_rate_segment3
        else:
            zc_ratio23 = np.nan
            zc_ratio13 = np.nan

        features[f"{prefix}ZC_RateRatio12"] = zc_ratio12
        features[f"{prefix}ZC_RateRatio23"] = zc_ratio23
        features[f"{prefix}ZC_RateRatio13"] = zc_ratio13
    else:
        features[f"{prefix}ZC_RateSegment1"] = np.nan
        features[f"{prefix}ZC_RateSegment2"] = np.nan
        features[f"{prefix}ZC_RateSegment3"] = np.nan
        features[f"{prefix}ZC_RateRatio12"] = np.nan
        features[f"{prefix}ZC_RateRatio23"] = np.nan
        features[f"{prefix}ZC_RateRatio13"] = np.nan

    # 4) Frequency Band Zero Crossing Analysis
    # Apply bandpass filters to isolate frequency bands and compute ZC in each band
    if emg_signal.size >= 10 and SCIPY_SIGNAL_AVAILABLE:
        try:
            def bandpass_filter(data, lowcut, highcut, fs, order=4):
                nyq = 0.5 * fs
                low = lowcut / nyq
                high = highcut / nyq
                # Ensure frequencies are within valid range
                if low <= 0:
                    low = 0.01
                if high >= 1:
                    high = 0.99
                if low >= high:
                    return data  # Return original if invalid range
                b, a = butter(order, [low, high], btype='band')
                return filtfilt(b, a, data)

            # Define frequency bands relevant to EMG and fatigue
            bands = {
                "Low": (10, 50),  # Low frequency band
                "Mid": (50, 100),  # Mid frequency band
                "High": (100, 200)  # High frequency band
            }

            for band_name, (low_freq, high_freq) in bands.items():
                try:
                    # Apply bandpass filter
                    filtered_signal = bandpass_filter(emg_signal, low_freq, high_freq, sampling_rate)

                    # Compute zero crossings in filtered signal
                    band_zc_total = np.sum(np.diff(np.sign(filtered_signal)) != 0)
                    band_zc_rate = band_zc_total / duration_sec if duration_sec > 0 else np.nan

                    features[f"{prefix}ZC_{band_name}Band_Total"] = band_zc_total
                    features[f"{prefix}ZC_{band_name}Band_Rate"] = band_zc_rate
                except Exception:
                    features[f"{prefix}ZC_{band_name}Band_Total"] = np.nan
                    features[f"{prefix}ZC_{band_name}Band_Rate"] = np.nan

            # Calculate band ratios (useful for fatigue detection)
            if (features[f"{prefix}ZC_HighBand_Rate"] > 0 and
                    not np.isnan(features[f"{prefix}ZC_LowBand_Rate"]) and
                    not np.isnan(features[f"{prefix}ZC_HighBand_Rate"])):

                features[f"{prefix}ZC_LowToHighRatio"] = (
                        features[f"{prefix}ZC_LowBand_Rate"] /
                        features[f"{prefix}ZC_HighBand_Rate"]
                )
            else:
                features[f"{prefix}ZC_LowToHighRatio"] = np.nan
        except Exception:
            # If filtering fails, set all band features to NaN
            for band_name in ["Low", "Mid", "High"]:
                features[f"{prefix}ZC_{band_name}Band_Total"] = np.nan
                features[f"{prefix}ZC_{band_name}Band_Rate"] = np.nan
            features[f"{prefix}ZC_LowToHighRatio"] = np.nan
    else:
        for band_name in ["Low", "Mid", "High"]:
            features[f"{prefix}ZC_{band_name}Band_Total"] = np.nan
            features[f"{prefix}ZC_{band_name}Band_Rate"] = np.nan
        features[f"{prefix}ZC_LowToHighRatio"] = np.nan

    # 5) Zero Crossing Fatigue Indices
    # Calculate specific fatigue-related indices based on zero crossings

    # ZC Fatigue Index: ratio of early to late ZC rate
    # Typically decreases with fatigue as frequency content shifts
    if emg_signal.size >= 4:
        half_point = emg_signal.size // 2
        early_segment = emg_signal[:half_point]
        late_segment = emg_signal[half_point:]

        early_zc = np.sum(np.diff(np.sign(early_segment)) != 0)
        late_zc = np.sum(np.diff(np.sign(late_segment)) != 0)

        half_duration = half_point / sampling_rate
        early_zc_rate = early_zc / half_duration if half_duration > 0 else np.nan
        late_zc_rate = late_zc / half_duration if half_duration > 0 else np.nan

        if late_zc_rate > 0 and not np.isnan(early_zc_rate) and not np.isnan(late_zc_rate):
            zc_fatigue_index = early_zc_rate / late_zc_rate
        else:
            zc_fatigue_index = np.nan

        features[f"{prefix}ZC_FatigueIndex"] = zc_fatigue_index
        features[f"{prefix}ZC_EarlyRate"] = early_zc_rate
        features[f"{prefix}ZC_LateRate"] = late_zc_rate
    else:
        features[f"{prefix}ZC_FatigueIndex"] = np.nan
        features[f"{prefix}ZC_EarlyRate"] = np.nan
        features[f"{prefix}ZC_LateRate"] = np.nan

    # ZC Trend: linear slope of ZC rate over time windows
    if emg_signal.size >= 100:  # Need enough samples for meaningful windows
        window_size = min(100, emg_signal.size // 5)  # Use 5 windows minimum
        step_size = window_size // 2  # 50% overlap

        window_zc_rates = []
        window_times = []

        for i in range(0, emg_signal.size - window_size, step_size):
            window = emg_signal[i:i + window_size]
            window_zc = np.sum(np.diff(np.sign(window)) != 0)
            window_duration = window_size / sampling_rate
            window_zc_rate = window_zc / window_duration

            window_zc_rates.append(window_zc_rate)
            window_times.append(i / sampling_rate)  # Time at window start

        if len(window_zc_rates) >= 2:
            # Calculate linear trend of ZC rate over time
            try:
                zc_trend_slope, _ = np.polyfit(window_times, window_zc_rates, 1)
                features[f"{prefix}ZC_TrendSlope"] = zc_trend_slope

                # Normalized ZC rate change (as percentage of initial value)
                if window_zc_rates[0] > 0:
                    zc_rate_change = (window_zc_rates[-1] - window_zc_rates[0]) / window_zc_rates[0]
                    features[f"{prefix}ZC_RateChange"] = zc_rate_change
                else:
                    features[f"{prefix}ZC_RateChange"] = np.nan
            except Exception:
                features[f"{prefix}ZC_TrendSlope"] = np.nan
                features[f"{prefix}ZC_RateChange"] = np.nan
        else:
            features[f"{prefix}ZC_TrendSlope"] = np.nan
            features[f"{prefix}ZC_RateChange"] = np.nan
    else:
        features[f"{prefix}ZC_TrendSlope"] = np.nan
        features[f"{prefix}ZC_RateChange"] = np.nan

    return features


def compute_emg_zc_features_multi_channel(
        emg_data: np.ndarray,
        sampling_rate: float,
        muscle_names: Optional[list] = None,
        threshold: float = 0.01
) -> Dict[str, float]:
    """
    Compute ZC features for multi-channel EMG data.

    Parameters:
    -----------
    emg_data : np.ndarray
        EMG data shaped (n_channels, n_samples)
    sampling_rate : float
        Sampling rate of the EMG data in Hz
    muscle_names : list, optional
        List of muscle names for each channel
    threshold : float, default 0.01
        Threshold to reduce noise sensitivity

    Returns:
    --------
    Dict[str, float]
        Dictionary of ZC features for all channels
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

        # Compute ZC features for this channel
        channel_features = compute_emg_zc_features_for_signal(
            channel_data, sampling_rate, muscle_name, threshold
        )
        features.update(channel_features)

    # Compute features for the mean across all channels (if multiple channels)
    if n_channels > 1:
        mean_emg = np.mean(emg_data, axis=0)
        mean_features = compute_emg_zc_features_for_signal(
            mean_emg, sampling_rate, "mean", threshold
        )
        features.update(mean_features)

    return features


def compute_emg_zc_features_for_window(
        window: Dict,
        fs_emg: float,
        muscle_names: Optional[list] = None,
        threshold: float = 0.01
) -> Dict[str, float]:
    """
    Convenience function to extract ZC EMG features from a window dict.

    Parameters:
    -----------
    window : Dict
        Window dictionary containing 'emg' key with shape (n_channels, n_samples)
    fs_emg : float
        EMG sampling frequency
    muscle_names : list, optional
        List of muscle names for each channel
    threshold : float, default 0.01
        Threshold to reduce noise sensitivity

    Returns:
    --------
    Dict[str, float]
        Dictionary of ZC EMG features
    """
    if "emg" not in window or window["emg"] is None:
        return {}

    emg_data = np.asarray(window["emg"])

    # Handle single channel case
    if emg_data.ndim == 1:
        emg_data = emg_data[np.newaxis, :]  # Add channel dimension

    return compute_emg_zc_features_multi_channel(
        emg_data, fs_emg, muscle_names, threshold
    )


def compute_emg_zc_features_per_repetition(
        emg_data: np.ndarray,
        repetition_segments: list,
        sampling_rate: float,
        muscle_names: Optional[list] = None,
        threshold: float = 0.01
) -> Dict[int, Dict[str, float]]:
    """
    Compute ZC EMG features for each repetition segment.

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
    threshold : float, default 0.01
        Threshold to reduce noise sensitivity

    Returns:
    --------
    Dict[int, Dict[str, float]]
        Dictionary mapping repetition number to ZC features
    """
    repetition_features = {}

    for rep_idx, (start_idx, end_idx) in enumerate(repetition_segments):
        # Extract data for this repetition
        rep_data = emg_data[:, start_idx:end_idx]

        if rep_data.shape[1] > 1:  # Need at least 2 samples for zero crossings
            # Compute ZC features for this repetition
            rep_features = compute_emg_zc_features_multi_channel(
                rep_data, sampling_rate, muscle_names, threshold
            )

            # Add repetition metadata
            rep_features["rep_duration"] = (end_idx - start_idx) / sampling_rate
            rep_features["rep_start_time"] = start_idx / sampling_rate
            rep_features["rep_end_time"] = end_idx / sampling_rate
            rep_features["rep_sample_count"] = end_idx - start_idx

            repetition_features[rep_idx + 1] = rep_features
        else:
            # Not enough data for ZC analysis
            repetition_features[rep_idx + 1] = {}

    return repetition_features

