"""
Mean Absolute Value (MAV) feature extraction for EMG signals.
Includes basic statistical features, fatigue-specific indicators, advanced metrics,
entropy measures, spectral analysis, and segmented analysis for comprehensive
muscle activation assessment.
"""

import numpy as np
from scipy.stats import skew, kurtosis
from typing import Dict, Optional

# Optional imports for advanced features
try:
    from scipy import signal

    SCIPY_SIGNAL_AVAILABLE = True
except ImportError:
    SCIPY_SIGNAL_AVAILABLE = False

try:
    import nolds

    NOLDS_AVAILABLE = True
except ImportError:
    NOLDS_AVAILABLE = False


def compute_sample_entropy_simple(data: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """
    Simple implementation of sample entropy when nolds is not available.

    Parameters:
    -----------
    data : np.ndarray
        Input signal
    m : int, default 2
        Pattern length
    r : float, default 0.2
        Tolerance for matching (relative to std)

    Returns:
    --------
    float
        Sample entropy value
    """
    if len(data) < m + 2:
        return np.nan

    # Normalize tolerance
    r = r * np.std(data) if r < 1 else r

    def _maxdist(xi, xj, N):
        return max([abs(ua - va) for ua, va in zip(xi, xj)])

    def _phi(m):
        patterns = np.array([data[i:i + m] for i in range(len(data) - m + 1)])
        C = np.zeros(len(patterns))
        for i in range(len(patterns)):
            template = patterns[i]
            for j in range(len(patterns)):
                if _maxdist(template, patterns[j], m) <= r:
                    C[i] += 1.0
        phi = (C / len(patterns)).mean()
        return phi

    try:
        return -np.log(_phi(m + 1) / _phi(m))
    except:
        return np.nan


def compute_approximate_entropy_simple(data: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """
    Simple implementation of approximate entropy when nolds is not available.

    Parameters:
    -----------
    data : np.ndarray
        Input signal
    m : int, default 2
        Pattern length
    r : float, default 0.2
        Tolerance for matching (relative to std)

    Returns:
    --------
    float
        Approximate entropy value
    """
    if len(data) < m + 1:
        return np.nan

    # Normalize tolerance
    r = r * np.std(data) if r < 1 else r

    def _maxdist(xi, xj):
        return max([abs(ua - va) for ua, va in zip(xi, xj)])

    def _phi(m):
        patterns = np.array([data[i:i + m] for i in range(len(data) - m + 1)])
        C = []
        for i in range(len(patterns)):
            template = patterns[i]
            matches = sum(1 for pattern in patterns if _maxdist(template, pattern) <= r)
            C.append(matches / len(patterns))
        phi = sum(np.log(c) for c in C if c > 0) / len(C)
        return phi

    try:
        return _phi(m) - _phi(m + 1)
    except:
        return np.nan


def compute_mav_features_for_signal(
        emg_signal: np.ndarray,
        sampling_rate: float,
        muscle_name: Optional[str] = None
) -> Dict[str, float]:
    """
    Compute comprehensive MAV features for a single EMG signal.

    Parameters:
    -----------
    emg_signal : np.ndarray
        1D array of EMG values
    sampling_rate : float
        Sampling rate of the EMG data in Hz
    muscle_name : str, optional
        Name of the muscle for feature naming

    Returns:
    --------
    Dict[str, float]
        Dictionary of MAV features
    """
    features = {}
    prefix = f"{muscle_name}_" if muscle_name else ""

    # Safety check
    if emg_signal.size == 0 or np.all(np.isnan(emg_signal)):
        # Return NaNs for all features if empty or all NaN
        feature_names = [
            # Original features
            "MAV_Total", "MAV_Per_Sec", "MAV_MinMaxNorm", "MAV_ZScoreNorm",
            "MAV_Slope", "MAV_SD", "MAV_CV", "MAV_Median", "MAV_IQR",
            "MAV_Max", "MAV_TimeToPeak", "MAV_Cumulative", "MAV_RiseTime",
            "MAV_DecayTime",
            # MAV Fatigue Indicators
            "MAV_FatigueIndex", "MAV_BurstCount", "MAV_BurstMeanDuration",
            "MAV_BurstMeanAmplitude", "MAV_RiseDecaySymmetry",
            # Advanced MAV Metrics
            "MAV_SampleEntropy", "MAV_ApproxEntropy", "MAV_SpectralCentroid",
            "MAV_SpectralSpread", "MAV_LowHighFreqRatio", "MAV_FatiguePattern",
            # Segmented MAV Analysis
            "MAV_Phase1", "MAV_Phase2", "MAV_Phase3", "MAV_PhaseRatio12",
            "MAV_PhaseRatio23", "MAV_PhaseRatio13",
            # MAV-based Fatigue Indices
            "MAV_DimitrovIndex", "MAV_EfficiencyIndex", "MAV_FatigueSlope"
        ]
        for feature_name in feature_names:
            features[f"{prefix}{feature_name}"] = np.nan
        return features

    # Clean input data - replace NaNs with zeros
    emg_signal = np.nan_to_num(emg_signal, nan=0.0)

    # 1. Basic MAV
    abs_emg = np.abs(emg_signal)
    n_samples = len(emg_signal)
    duration = n_samples / sampling_rate

    # (a) Total MAV = sum(abs(EMG))
    mav_total = np.sum(abs_emg)
    features[f"{prefix}MAV_Total"] = float(mav_total)

    # (b) MAV per second = MAV_Total / repetition duration
    mav_per_sec = mav_total / duration if duration > 0 else np.nan
    features[f"{prefix}MAV_Per_Sec"] = float(mav_per_sec)

    # 2. Normalized MAV
    # (a) Min-Max Normalized (0..1) across this signal
    min_val = np.min(abs_emg)
    max_val = np.max(abs_emg)
    if max_val > min_val:
        norm_minmax = (abs_emg - min_val) / (max_val - min_val)
    else:
        # all values might be identical
        norm_minmax = np.zeros_like(abs_emg)
    # Store the mean of the min-max normalized signal
    features[f"{prefix}MAV_MinMaxNorm"] = float(np.mean(norm_minmax))

    # (b) Z-score Normalized => (abs_emg - mean) / std
    mean_val = np.mean(abs_emg)
    std_val = np.std(abs_emg, ddof=1) if n_samples > 1 else 0.0
    if std_val > 1e-12:
        zscore_vals = (abs_emg - mean_val) / std_val
    else:
        zscore_vals = np.zeros_like(abs_emg)
    features[f"{prefix}MAV_ZScoreNorm"] = float(np.mean(zscore_vals))

    # 3. MAV Slope => linear slope of abs_emg(t) over time
    t = np.arange(n_samples) / sampling_rate
    if n_samples > 1:
        slope_val = np.polyfit(t, abs_emg, 1)[0]  # slope coefficient
        features[f"{prefix}MAV_Slope"] = float(slope_val)
    else:
        features[f"{prefix}MAV_Slope"] = np.nan

    # 4. MAV Variability
    # (a) Standard Deviation
    features[f"{prefix}MAV_SD"] = float(std_val)
    # (b) Coefficient of Variation (CV = SD / mean(MAV))
    if mean_val > 1e-12:
        cv_val = std_val / mean_val
    else:
        cv_val = np.nan
    features[f"{prefix}MAV_CV"] = float(cv_val)

    # 5. MAV Percentiles
    mav_median = np.median(abs_emg)
    features[f"{prefix}MAV_Median"] = float(mav_median)
    q1 = np.percentile(abs_emg, 25)
    q3 = np.percentile(abs_emg, 75)
    features[f"{prefix}MAV_IQR"] = float(q3 - q1)

    # 6. MAV Peaks
    # (a) Maximum MAV (peak)
    mav_max = max_val
    features[f"{prefix}MAV_Max"] = float(mav_max)
    # (b) Time to Peak MAV
    idx_peak = np.argmax(abs_emg)
    time_to_peak = idx_peak / sampling_rate
    features[f"{prefix}MAV_TimeToPeak"] = float(time_to_peak)

    # 7. Temporal MAV Patterns
    # (a) Cumulative MAV => final sum is the same as mav_total
    features[f"{prefix}MAV_Cumulative"] = float(mav_total)

    # (b) MAV Rise Time => how long does it take to go from "baseline" to peak?
    # Define baseline as 10% of peak
    if mav_max > 1e-12:
        threshold_rise = 0.1 * mav_max
        rise_idx = np.where(abs_emg >= threshold_rise)[0]
        if len(rise_idx) > 0:
            mav_rise_time = (rise_idx[0]) / sampling_rate
        else:
            mav_rise_time = np.nan
    else:
        mav_rise_time = np.nan
    features[f"{prefix}MAV_RiseTime"] = float(mav_rise_time)

    # (c) MAV Decay Time => from peak back to baseline
    if mav_max > 1e-12:
        threshold_decay = 0.1 * mav_max
        post_peak = abs_emg[idx_peak:]
        decay_idx = np.where(post_peak <= threshold_decay)[0]
        if len(decay_idx) > 0:
            # time from peak to decay
            mav_decay_time = decay_idx[0] / sampling_rate
        else:
            mav_decay_time = np.nan
    else:
        mav_decay_time = np.nan
    features[f"{prefix}MAV_DecayTime"] = float(mav_decay_time)

    # === MAV FATIGUE INDICATORS ===

    # 1. MAV Fatigue Index (ratio of MAV in first vs. last part)
    if n_samples >= 4:  # Need at least 4 samples
        half_point = n_samples // 2
        first_half = abs_emg[:half_point]
        last_half = abs_emg[half_point:]

        if len(first_half) > 0 and len(last_half) > 0:
            mav_first_half = np.mean(first_half)
            mav_last_half = np.mean(last_half)

            # Fatigue index: ratio of first half to last half
            if mav_last_half > 1e-12:
                fatigue_index = mav_first_half / mav_last_half
            else:
                fatigue_index = np.nan

            features[f"{prefix}MAV_FatigueIndex"] = float(fatigue_index)
        else:
            features[f"{prefix}MAV_FatigueIndex"] = np.nan
    else:
        features[f"{prefix}MAV_FatigueIndex"] = np.nan

    # 2. MAV Burst Analysis
    if n_samples >= 10 and mav_max > 1e-12:
        try:
            # Define burst threshold as 30% of max MAV
            burst_threshold = 0.3 * mav_max

            # Find regions where MAV exceeds threshold
            above_threshold = abs_emg > burst_threshold

            # Find burst starts and ends
            burst_starts = np.where(np.diff(above_threshold.astype(int)) == 1)[0] + 1
            burst_ends = np.where(np.diff(above_threshold.astype(int)) == -1)[0] + 1

            # Handle edge cases
            if above_threshold[0]:
                burst_starts = np.insert(burst_starts, 0, 0)
            if above_threshold[-1]:
                burst_ends = np.append(burst_ends, n_samples - 1)

            # Calculate burst metrics
            if len(burst_starts) > 0 and len(burst_ends) > 0 and len(burst_starts) == len(burst_ends):
                burst_count = len(burst_starts)
                burst_durations = [(burst_ends[i] - burst_starts[i]) / sampling_rate for i in range(burst_count)]

                # Calculate burst amplitudes
                burst_amplitudes = []
                for i in range(burst_count):
                    start_idx = burst_starts[i]
                    end_idx = burst_ends[i]
                    if start_idx < end_idx and start_idx < len(abs_emg) and end_idx <= len(abs_emg):
                        burst_segment = abs_emg[start_idx:end_idx]
                        if len(burst_segment) > 0:
                            burst_amplitudes.append(np.mean(burst_segment))

                features[f"{prefix}MAV_BurstCount"] = float(burst_count)
                features[f"{prefix}MAV_BurstMeanDuration"] = float(
                    np.mean(burst_durations)) if burst_durations else np.nan
                features[f"{prefix}MAV_BurstMeanAmplitude"] = float(
                    np.mean(burst_amplitudes)) if burst_amplitudes else np.nan
            else:
                features[f"{prefix}MAV_BurstCount"] = 0.0
                features[f"{prefix}MAV_BurstMeanDuration"] = np.nan
                features[f"{prefix}MAV_BurstMeanAmplitude"] = np.nan
        except Exception:
            features[f"{prefix}MAV_BurstCount"] = np.nan
            features[f"{prefix}MAV_BurstMeanDuration"] = np.nan
            features[f"{prefix}MAV_BurstMeanAmplitude"] = np.nan
    else:
        features[f"{prefix}MAV_BurstCount"] = np.nan
        features[f"{prefix}MAV_BurstMeanDuration"] = np.nan
        features[f"{prefix}MAV_BurstMeanAmplitude"] = np.nan

    # 3. MAV Rise/Decay Symmetry
    if not np.isnan(mav_rise_time) and not np.isnan(mav_decay_time) and mav_decay_time > 1e-12:
        rise_decay_symmetry = mav_rise_time / mav_decay_time
        features[f"{prefix}MAV_RiseDecaySymmetry"] = float(rise_decay_symmetry)
    else:
        features[f"{prefix}MAV_RiseDecaySymmetry"] = np.nan

    # === ADVANCED MAV METRICS ===

    # 1. MAV Complexity (sample entropy and approximate entropy)
    if n_samples >= 100:
        try:
            if NOLDS_AVAILABLE:
                # Use nolds library for entropy calculations
                sample_entropy = nolds.sampen(abs_emg)
                approx_entropy = nolds.hurst_rs(abs_emg) if len(abs_emg) > 20 else np.nan
            else:
                # Use simple implementations
                sample_entropy = compute_sample_entropy_simple(abs_emg, m=2, r=0.2)
                approx_entropy = compute_approximate_entropy_simple(abs_emg, m=2, r=0.2)

            features[f"{prefix}MAV_SampleEntropy"] = float(sample_entropy)
            features[f"{prefix}MAV_ApproxEntropy"] = float(approx_entropy)
        except Exception:
            # Fallback: simple complexity measure
            if n_samples >= 10:
                try:
                    path_length = np.sum(np.abs(np.diff(abs_emg)))
                    direct_path = np.abs(abs_emg[-1] - abs_emg[0])
                    if direct_path > 1e-12:
                        complexity = path_length / direct_path
                    else:
                        complexity = np.nan

                    features[f"{prefix}MAV_SampleEntropy"] = float(complexity)
                    features[f"{prefix}MAV_ApproxEntropy"] = np.nan
                except Exception:
                    features[f"{prefix}MAV_SampleEntropy"] = np.nan
                    features[f"{prefix}MAV_ApproxEntropy"] = np.nan
            else:
                features[f"{prefix}MAV_SampleEntropy"] = np.nan
                features[f"{prefix}MAV_ApproxEntropy"] = np.nan
    else:
        features[f"{prefix}MAV_SampleEntropy"] = np.nan
        features[f"{prefix}MAV_ApproxEntropy"] = np.nan

    # 2. MAV Frequency Content (spectral analysis)
    if n_samples >= 10 and SCIPY_SIGNAL_AVAILABLE:
        try:
            # Detrend MAV signal
            detrended_mav = signal.detrend(abs_emg)

            # Calculate power spectral density
            f, psd = signal.welch(detrended_mav, fs=sampling_rate, nperseg=min(256, n_samples))

            if len(psd) > 0 and np.sum(psd) > 1e-12:
                # Normalize PSD
                norm_psd = psd / np.sum(psd)

                # Spectral centroid
                spectral_centroid = np.sum(f * norm_psd)
                features[f"{prefix}MAV_SpectralCentroid"] = float(spectral_centroid)

                # Spectral spread
                spectral_spread = np.sqrt(np.sum(((f - spectral_centroid) ** 2) * norm_psd))
                features[f"{prefix}MAV_SpectralSpread"] = float(spectral_spread)

                # Low/High frequency ratio
                low_freq_idx = f <= 5  # Low frequency components (< 5 Hz)
                high_freq_idx = f > 5  # High frequency components (> 5 Hz)

                low_freq_power = np.sum(psd[low_freq_idx]) if np.any(low_freq_idx) else 0
                high_freq_power = np.sum(psd[high_freq_idx]) if np.any(high_freq_idx) else 0

                if high_freq_power > 1e-12:
                    low_high_ratio = low_freq_power / high_freq_power
                else:
                    low_high_ratio = np.nan

                features[f"{prefix}MAV_LowHighFreqRatio"] = float(low_high_ratio)
            else:
                features[f"{prefix}MAV_SpectralCentroid"] = np.nan
                features[f"{prefix}MAV_SpectralSpread"] = np.nan
                features[f"{prefix}MAV_LowHighFreqRatio"] = np.nan
        except Exception:
            features[f"{prefix}MAV_SpectralCentroid"] = np.nan
            features[f"{prefix}MAV_SpectralSpread"] = np.nan
            features[f"{prefix}MAV_LowHighFreqRatio"] = np.nan
    else:
        features[f"{prefix}MAV_SpectralCentroid"] = np.nan
        features[f"{prefix}MAV_SpectralSpread"] = np.nan
        features[f"{prefix}MAV_LowHighFreqRatio"] = np.nan

    # 3. MAV Pattern Recognition
    if n_samples >= 20:
        try:
            # Calculate moving average to smooth the signal
            window_size = min(5, n_samples // 4)
            if window_size > 0:
                smoothed = np.convolve(abs_emg, np.ones(window_size) / window_size, mode='valid')

                if len(smoothed) > 1:
                    slopes = np.diff(smoothed)

                    # Count slope sign changes
                    sign_changes = np.sum(np.diff(np.signbit(slopes)))

                    # Normalize by signal length
                    pattern_complexity = sign_changes / (len(slopes) - 1) if len(slopes) > 1 else np.nan

                    features[f"{prefix}MAV_FatiguePattern"] = float(pattern_complexity)
                else:
                    features[f"{prefix}MAV_FatiguePattern"] = np.nan
            else:
                features[f"{prefix}MAV_FatiguePattern"] = np.nan
        except Exception:
            features[f"{prefix}MAV_FatiguePattern"] = np.nan
    else:
        features[f"{prefix}MAV_FatiguePattern"] = np.nan

    # === SEGMENTED MAV ANALYSIS ===

    if n_samples >= 6:  # Need enough samples for meaningful segmentation
        try:
            # Divide into three equal parts
            third_point = n_samples // 3
            two_thirds_point = 2 * (n_samples // 3)

            if third_point > 0 and two_thirds_point > third_point and n_samples > two_thirds_point:
                phase1 = np.mean(abs_emg[:third_point])
                phase2 = np.mean(abs_emg[third_point:two_thirds_point])
                phase3 = np.mean(abs_emg[two_thirds_point:])

                features[f"{prefix}MAV_Phase1"] = float(phase1)
                features[f"{prefix}MAV_Phase2"] = float(phase2)
                features[f"{prefix}MAV_Phase3"] = float(phase3)

                # Calculate phase ratios
                if phase2 > 1e-12:
                    phase_ratio12 = phase1 / phase2
                else:
                    phase_ratio12 = np.nan

                if phase3 > 1e-12:
                    phase_ratio23 = phase2 / phase3
                    phase_ratio13 = phase1 / phase3
                else:
                    phase_ratio23 = np.nan
                    phase_ratio13 = np.nan

                features[f"{prefix}MAV_PhaseRatio12"] = float(phase_ratio12)
                features[f"{prefix}MAV_PhaseRatio23"] = float(phase_ratio23)
                features[f"{prefix}MAV_PhaseRatio13"] = float(phase_ratio13)
            else:
                features[f"{prefix}MAV_Phase1"] = np.nan
                features[f"{prefix}MAV_Phase2"] = np.nan
                features[f"{prefix}MAV_Phase3"] = np.nan
                features[f"{prefix}MAV_PhaseRatio12"] = np.nan
                features[f"{prefix}MAV_PhaseRatio23"] = np.nan
                features[f"{prefix}MAV_PhaseRatio13"] = np.nan
        except Exception:
            features[f"{prefix}MAV_Phase1"] = np.nan
            features[f"{prefix}MAV_Phase2"] = np.nan
            features[f"{prefix}MAV_Phase3"] = np.nan
            features[f"{prefix}MAV_PhaseRatio12"] = np.nan
            features[f"{prefix}MAV_PhaseRatio23"] = np.nan
            features[f"{prefix}MAV_PhaseRatio13"] = np.nan
    else:
        features[f"{prefix}MAV_Phase1"] = np.nan
        features[f"{prefix}MAV_Phase2"] = np.nan
        features[f"{prefix}MAV_Phase3"] = np.nan
        features[f"{prefix}MAV_PhaseRatio12"] = np.nan
        features[f"{prefix}MAV_PhaseRatio23"] = np.nan
        features[f"{prefix}MAV_PhaseRatio13"] = np.nan

    # === MAV-BASED FATIGUE INDICES ===

    # 1. Dimitrov Index (modified for MAV)
    if n_samples >= 10:
        try:
            if mean_val > 1e-12:
                dimitrov_index = std_val / mean_val
            else:
                dimitrov_index = np.nan

            features[f"{prefix}MAV_DimitrovIndex"] = float(dimitrov_index)
        except Exception:
            features[f"{prefix}MAV_DimitrovIndex"] = np.nan
    else:
        features[f"{prefix}MAV_DimitrovIndex"] = np.nan

    # 2. MAV Efficiency Index
    if n_samples >= 10 and mav_max > 1e-12:
        try:
            efficiency_index = mean_val / mav_max
            features[f"{prefix}MAV_EfficiencyIndex"] = float(efficiency_index)
        except Exception:
            features[f"{prefix}MAV_EfficiencyIndex"] = np.nan
    else:
        features[f"{prefix}MAV_EfficiencyIndex"] = np.nan

    # 3. MAV Fatigue Slope (slope in last third)
    if n_samples >= 10:
        try:
            two_thirds_point = 2 * (n_samples // 3)
            if two_thirds_point < n_samples:
                last_third = abs_emg[two_thirds_point:]
                t_last_third = t[two_thirds_point:]

                if len(last_third) > 1:
                    fatigue_slope = np.polyfit(t_last_third, last_third, 1)[0]
                else:
                    fatigue_slope = np.nan

                features[f"{prefix}MAV_FatigueSlope"] = float(fatigue_slope)
            else:
                features[f"{prefix}MAV_FatigueSlope"] = np.nan
        except Exception:
            features[f"{prefix}MAV_FatigueSlope"] = np.nan
    else:
        features[f"{prefix}MAV_FatigueSlope"] = np.nan

    return features


def compute_emg_mav_features_multi_channel(
        emg_data: np.ndarray,
        sampling_rate: float,
        muscle_names: Optional[list] = None
) -> Dict[str, float]:
    """
    Compute MAV features for multi-channel EMG data.

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
        Dictionary of MAV features for all channels
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

        # Compute MAV features for this channel
        channel_features = compute_mav_features_for_signal(
            channel_data, sampling_rate, muscle_name
        )
        features.update(channel_features)

    # Compute features for the mean across all channels (if multiple channels)
    if n_channels > 1:
        mean_emg = np.mean(emg_data, axis=0)
        mean_features = compute_mav_features_for_signal(
            mean_emg, sampling_rate, "mean"
        )
        features.update(mean_features)

    return features


def compute_emg_mav_features_for_window(
        window: Dict,
        fs_emg: float,
        muscle_names: Optional[list] = None
) -> Dict[str, float]:
    """
    Convenience function to extract MAV features from a window dict.

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
        Dictionary of MAV features
    """
    if "emg" not in window or window["emg"] is None:
        return {}

    emg_data = np.asarray(window["emg"])

    # Handle single channel case
    if emg_data.ndim == 1:
        emg_data = emg_data[np.newaxis, :]  # Add channel dimension

    return compute_emg_mav_features_multi_channel(
        emg_data, fs_emg, muscle_names
    )


def compute_emg_mav_features_per_repetition(
        emg_data: np.ndarray,
        repetition_segments: list,
        sampling_rate: float,
        muscle_names: Optional[list] = None
) -> Dict[int, Dict[str, float]]:
    """
    Compute MAV features for each repetition segment.

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
        Dictionary mapping repetition number to MAV features
    """
    repetition_features = {}

    for rep_idx, (start_idx, end_idx) in enumerate(repetition_segments):
        # Extract data for this repetition
        rep_data = emg_data[:, start_idx:end_idx]

        if rep_data.shape[1] > 0:  # Ensure we have data
            # Compute MAV features for this repetition
            rep_features = compute_emg_mav_features_multi_channel(
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
