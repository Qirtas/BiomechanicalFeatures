"""
Short-Time Fourier Transform (STFT) feature extraction for EMG signals.
Includes comprehensive time-frequency analysis, fatigue-specific indicators,
spectral characteristics, and advanced STFT-based features for muscle activation assessment.
"""

import numpy as np
from scipy.stats import skew, kurtosis
from typing import Dict, Optional

# Optional imports for advanced features
try:
    from scipy.signal import stft, savgol_filter

    SCIPY_SIGNAL_AVAILABLE = True
except ImportError:
    SCIPY_SIGNAL_AVAILABLE = False


def compute_stft_features_for_signal(
        emg_signal: np.ndarray,
        sampling_rate: float = 1000,
        muscle_name: Optional[str] = None,
        window_size: int = 256,
        overlap: int = 128
) -> Dict[str, float]:
    """
    Compute comprehensive STFT-based features for a single EMG signal.

    Parameters:
    -----------
    emg_signal : np.ndarray
        1D array of EMG values
    sampling_rate : float, default 1000
        Sampling rate of the EMG signal in Hz
    muscle_name : str, optional
        Name of the muscle for feature naming
    window_size : int, default 256
        Size of the STFT window
    overlap : int, default 128
        Overlap between consecutive windows

    Returns:
    --------
    Dict[str, float]
        Dictionary of STFT-based features
    """
    features = {}
    prefix = f"{muscle_name}_" if muscle_name else ""

    # Safety check
    if emg_signal.size == 0 or np.all(np.isnan(emg_signal)):
        placeholders = [
            "STFT_MNF", "STFT_MDF", "STFT_PeakFreq", "STFT_TotalPower",
            "STFT_SpectralEntropy", "STFT_LowFreqPower", "STFT_HighFreqPower",
            "STFT_FreqBandRatio", "STFT_SpectralSlope", "STFT_MNF_Std",
            "STFT_MDF_Std", "STFT_Skewness", "STFT_DimitrovIndex",
            "STFT_SpectralMoment2", "STFT_SpectralMoment3", "STFT_MNF_Slope",
            "STFT_MNF_Decrease", "STFT_Bandwidth", "STFT_PeakFreqVariability",
            "STFT_VeryLowPower", "STFT_LowPower", "STFT_MidPower",
            "STFT_HighPower", "STFT_VeryHighPower", "STFT_LowToVeryLowRatio",
            "STFT_MidToVeryLowRatio", "STFT_HighToVeryLowRatio", "STFT_VeryHighToVeryLowRatio",
            "STFT_EarlyMNF", "STFT_MiddleMNF", "STFT_LateMNF", "STFT_MNF_FatigueIndex"
        ]
        for pl in placeholders:
            features[f"{prefix}{pl}"] = np.nan
        return features

    # Clean input data
    emg_signal = np.nan_to_num(emg_signal, nan=0.0)

    # Check if we have scipy.signal for STFT
    if not SCIPY_SIGNAL_AVAILABLE:
        # Fallback: return NaN for all features if scipy not available
        placeholders = [
            "STFT_MNF", "STFT_MDF", "STFT_PeakFreq", "STFT_TotalPower",
            "STFT_SpectralEntropy", "STFT_LowFreqPower", "STFT_HighFreqPower",
            "STFT_FreqBandRatio", "STFT_SpectralSlope", "STFT_MNF_Std",
            "STFT_MDF_Std", "STFT_Skewness", "STFT_DimitrovIndex",
            "STFT_SpectralMoment2", "STFT_SpectralMoment3", "STFT_MNF_Slope",
            "STFT_MNF_Decrease", "STFT_Bandwidth", "STFT_PeakFreqVariability",
            "STFT_VeryLowPower", "STFT_LowPower", "STFT_MidPower",
            "STFT_HighPower", "STFT_VeryHighPower", "STFT_LowToVeryLowRatio",
            "STFT_MidToVeryLowRatio", "STFT_HighToVeryLowRatio", "STFT_VeryHighToVeryLowRatio",
            "STFT_EarlyMNF", "STFT_MiddleMNF", "STFT_LateMNF", "STFT_MNF_FatigueIndex"
        ]
        for pl in placeholders:
            features[f"{prefix}{pl}"] = np.nan
        return features

    try:
        # Perform STFT
        f, t, Zxx = stft(emg_signal, fs=sampling_rate, nperseg=window_size, noverlap=overlap)
        power_spectrum = np.abs(Zxx) ** 2  # Power spectrum

        # Handle empty power spectrum
        if power_spectrum.size == 0 or np.all(np.isnan(power_spectrum)):
            placeholders = [
                "STFT_MNF", "STFT_MDF", "STFT_PeakFreq", "STFT_TotalPower",
                "STFT_SpectralEntropy", "STFT_LowFreqPower", "STFT_HighFreqPower",
                "STFT_FreqBandRatio", "STFT_SpectralSlope", "STFT_MNF_Std",
                "STFT_MDF_Std", "STFT_Skewness", "STFT_DimitrovIndex",
                "STFT_SpectralMoment2", "STFT_SpectralMoment3", "STFT_MNF_Slope",
                "STFT_MNF_Decrease", "STFT_Bandwidth", "STFT_PeakFreqVariability",
                "STFT_VeryLowPower", "STFT_LowPower", "STFT_MidPower",
                "STFT_HighPower", "STFT_VeryHighPower", "STFT_LowToVeryLowRatio",
                "STFT_MidToVeryLowRatio", "STFT_HighToVeryLowRatio", "STFT_VeryHighToVeryLowRatio",
                "STFT_EarlyMNF", "STFT_MiddleMNF", "STFT_LateMNF", "STFT_MNF_FatigueIndex"
            ]
            for pl in placeholders:
                features[f"{prefix}{pl}"] = np.nan
            return features

        # Replace any NaN values with zeros
        power_spectrum = np.nan_to_num(power_spectrum, nan=0.0)

        # Average over time windows
        mean_power_spectrum = np.mean(power_spectrum, axis=1)  # Average over time windows

        # Safety check for all-zero power spectrum
        if np.sum(mean_power_spectrum) <= 1e-10:
            placeholders = [
                "STFT_MNF", "STFT_MDF", "STFT_PeakFreq", "STFT_TotalPower",
                "STFT_SpectralEntropy", "STFT_LowFreqPower", "STFT_HighFreqPower",
                "STFT_FreqBandRatio", "STFT_SpectralSlope", "STFT_MNF_Std",
                "STFT_MDF_Std", "STFT_Skewness", "STFT_DimitrovIndex",
                "STFT_SpectralMoment2", "STFT_SpectralMoment3", "STFT_MNF_Slope",
                "STFT_MNF_Decrease", "STFT_Bandwidth", "STFT_PeakFreqVariability",
                "STFT_VeryLowPower", "STFT_LowPower", "STFT_MidPower",
                "STFT_HighPower", "STFT_VeryHighPower", "STFT_LowToVeryLowRatio",
                "STFT_MidToVeryLowRatio", "STFT_HighToVeryLowRatio", "STFT_VeryHighToVeryLowRatio",
                "STFT_EarlyMNF", "STFT_MiddleMNF", "STFT_LateMNF", "STFT_MNF_FatigueIndex"
            ]
            for pl in placeholders:
                features[f"{prefix}{pl}"] = np.nan
            return features

        # 1) Mean Frequency (MNF)
        if np.sum(mean_power_spectrum) > 0:
            mnf = np.sum(f * mean_power_spectrum) / np.sum(mean_power_spectrum)
        else:
            mnf = np.nan
        features[f"{prefix}STFT_MNF"] = float(mnf)

        # 2) Median Frequency (MDF)
        if np.sum(mean_power_spectrum) > 0:
            cumulative_power = np.cumsum(mean_power_spectrum)
            if cumulative_power[-1] > 0:
                mdf_index = np.argmin(np.abs(cumulative_power - 0.5 * cumulative_power[-1]))
                mdf = f[mdf_index]
            else:
                mdf = np.nan
        else:
            mdf = np.nan
        features[f"{prefix}STFT_MDF"] = float(mdf)

        # 3) Peak Frequency
        if np.any(mean_power_spectrum > 0):
            peak_freq_index = np.argmax(mean_power_spectrum)
            peak_freq = f[peak_freq_index]
        else:
            peak_freq = np.nan
        features[f"{prefix}STFT_PeakFreq"] = float(peak_freq)

        # 4) Total Power
        total_power = np.sum(mean_power_spectrum)
        features[f"{prefix}STFT_TotalPower"] = float(total_power)

        # 5) Spectral Entropy
        if total_power > 0:
            normalized_power = mean_power_spectrum / total_power
            # Avoid log(0) by adding a small epsilon to zeros
            normalized_power = np.where(normalized_power > 0, normalized_power, 1e-10)
            try:
                spectral_entropy = -np.sum(normalized_power * np.log2(normalized_power))
            except:
                spectral_entropy = np.nan
        else:
            spectral_entropy = np.nan
        features[f"{prefix}STFT_SpectralEntropy"] = float(spectral_entropy)

        # 6) Frequency Band Power Ratios
        low_freq_mask = (f >= 0) & (f <= 50)
        high_freq_mask = (f > 50) & (f <= 500)

        # Check if masks have any True values
        if np.any(low_freq_mask):
            low_freq_power = np.sum(mean_power_spectrum[low_freq_mask])
        else:
            low_freq_power = 0

        if np.any(high_freq_mask):
            high_freq_power = np.sum(mean_power_spectrum[high_freq_mask])
        else:
            high_freq_power = 0

        freq_band_ratio = low_freq_power / high_freq_power if high_freq_power > 1e-10 else np.nan
        features[f"{prefix}STFT_LowFreqPower"] = float(low_freq_power)
        features[f"{prefix}STFT_HighFreqPower"] = float(high_freq_power)
        features[f"{prefix}STFT_FreqBandRatio"] = float(freq_band_ratio)

        # 7) Spectral Slope
        try:
            coeffs = np.polyfit(f, mean_power_spectrum, 1)
            spectral_slope = coeffs[0]  # Slope of the linear fit
        except:
            spectral_slope = np.nan
        features[f"{prefix}STFT_SpectralSlope"] = float(spectral_slope)

        # 8) Standard Deviation of MNF/MDF
        try:
            # Safe calculation of MNF for each time window
            mnf_values = []
            for i in range(power_spectrum.shape[1]):
                ps = power_spectrum[:, i]
                if np.sum(ps) > 1e-10:
                    mnf_values.append(np.sum(f * ps) / np.sum(ps))

            mnf_std = np.std(mnf_values, ddof=1) if len(mnf_values) > 1 else np.nan

            # Safe calculation of MDF for each time window
            mdf_values = []
            for i in range(power_spectrum.shape[1]):
                ps = power_spectrum[:, i]
                if np.sum(ps) > 1e-10:
                    cum_ps = np.cumsum(ps)
                    if cum_ps[-1] > 0:
                        mdf_idx = np.argmin(np.abs(cum_ps - 0.5 * cum_ps[-1]))
                        mdf_values.append(f[mdf_idx])

            mdf_std = np.std(mdf_values, ddof=1) if len(mdf_values) > 1 else np.nan
        except:
            mnf_std = np.nan
            mdf_std = np.nan

        features[f"{prefix}STFT_MNF_Std"] = float(mnf_std)
        features[f"{prefix}STFT_MDF_Std"] = float(mdf_std)

        # 9) Skewness of the Spectrum
        try:
            skewness = skew(mean_power_spectrum)
        except:
            skewness = np.nan
        features[f"{prefix}STFT_Skewness"] = float(skewness)

        # 10) Dimitrov Fatigue Index (modified for STFT)
        try:
            # Avoid division by zero by checking if f values are non-zero
            if np.any(low_freq_mask) and np.any(f[low_freq_mask] > 0):
                # Use only non-zero frequency values
                valid_low_mask = low_freq_mask & (f > 0)
                if np.any(valid_low_mask):
                    low_freq_weighted = np.sum(mean_power_spectrum[valid_low_mask] / f[valid_low_mask])
                else:
                    low_freq_weighted = np.nan
            else:
                low_freq_weighted = np.nan

            if np.any(high_freq_mask) and np.any(f[high_freq_mask] > 0):
                # Use only non-zero frequency values
                valid_high_mask = high_freq_mask & (f > 0)
                if np.any(valid_high_mask):
                    high_freq_weighted = np.sum(mean_power_spectrum[valid_high_mask] / f[valid_high_mask])
                else:
                    high_freq_weighted = np.nan
            else:
                high_freq_weighted = np.nan

            dimitrov_index = (low_freq_weighted / high_freq_weighted
                              if not np.isnan(low_freq_weighted) and not np.isnan(high_freq_weighted)
                                 and high_freq_weighted > 1e-10 else np.nan)
        except:
            dimitrov_index = np.nan

        features[f"{prefix}STFT_DimitrovIndex"] = float(dimitrov_index)

        # 11) Spectral Moments
        try:
            if np.sum(mean_power_spectrum) > 1e-10:
                # First moment is the same as MNF
                moment1 = mnf
                # Second moment (related to spectral spread)
                moment2 = np.sum((f - moment1) ** 2 * mean_power_spectrum) / np.sum(mean_power_spectrum)
                # Third moment (related to spectral skewness)
                moment3 = np.sum((f - moment1) ** 3 * mean_power_spectrum) / np.sum(mean_power_spectrum)
            else:
                moment2 = np.nan
                moment3 = np.nan
        except:
            moment2 = np.nan
            moment3 = np.nan

        features[f"{prefix}STFT_SpectralMoment2"] = float(moment2)
        features[f"{prefix}STFT_SpectralMoment3"] = float(moment3)

        # 12) Fatigue Trend Features
        try:
            # Calculate MNF for each time window to track fatigue progression
            mnf_per_window = []
            for i in range(power_spectrum.shape[1]):
                ps = power_spectrum[:, i]
                if np.sum(ps) > 1e-10:
                    mnf_per_window.append(np.sum(f * ps) / np.sum(ps))

            if len(mnf_per_window) >= 2:
                # Linear regression slope of MNF over time (negative indicates fatigue)
                x_vals = np.arange(len(mnf_per_window))
                mnf_slope, _ = np.polyfit(x_vals, mnf_per_window, 1)
                # Normalized MNF decrease (as percentage of initial value)
                if mnf_per_window[0] > 1e-10:
                    mnf_decrease = (mnf_per_window[0] - mnf_per_window[-1]) / mnf_per_window[0]
                else:
                    mnf_decrease = np.nan
            else:
                mnf_slope = np.nan
                mnf_decrease = np.nan
        except:
            mnf_slope = np.nan
            mnf_decrease = np.nan

        features[f"{prefix}STFT_MNF_Slope"] = float(mnf_slope)
        features[f"{prefix}STFT_MNF_Decrease"] = float(mnf_decrease)

        # 13) Spectral Bandwidth
        try:
            if total_power > 1e-10:
                # Width of the frequency band containing most of the power
                power_threshold = 0.8  # 80% of total power
                sorted_power = np.sort(mean_power_spectrum)[::-1]  # Sort in descending order
                cumulative_sorted = np.cumsum(sorted_power) / total_power

                if np.any(cumulative_sorted >= power_threshold):
                    bandwidth_idx = np.argmax(cumulative_sorted >= power_threshold)
                    spectral_bandwidth = bandwidth_idx / len(f) * (f[-1] - f[0])
                else:
                    spectral_bandwidth = np.nan
            else:
                spectral_bandwidth = np.nan
        except:
            spectral_bandwidth = np.nan

        features[f"{prefix}STFT_Bandwidth"] = float(spectral_bandwidth)

        # 14) Instantaneous Frequency Variability
        try:
            # Variability in the dominant frequency over time
            peak_freqs = []
            for i in range(power_spectrum.shape[1]):
                ps = power_spectrum[:, i]
                if np.any(ps > 0):
                    peak_freqs.append(f[np.argmax(ps)])

            peak_freq_variability = np.std(peak_freqs, ddof=1) if len(peak_freqs) > 1 else np.nan
        except:
            peak_freq_variability = np.nan

        features[f"{prefix}STFT_PeakFreqVariability"] = float(peak_freq_variability)

        # 15) Multi-band power analysis
        try:
            # Divide spectrum into more specific frequency bands relevant to EMG
            bands = {
                "VeryLow": (0, 20),  # Very low frequency components
                "Low": (20, 50),  # Low frequency components
                "Mid": (50, 100),  # Mid frequency components
                "High": (100, 200),  # High frequency components
                "VeryHigh": (200, 500)  # Very high frequency components
            }

            band_powers = {}
            for band_name, (low, high) in bands.items():
                band_mask = (f >= low) & (f <= high)
                if np.any(band_mask):
                    band_power = np.sum(mean_power_spectrum[band_mask])
                else:
                    band_power = 0

                features[f"{prefix}STFT_{band_name}Power"] = float(band_power)
                band_powers[band_name] = band_power

            # Calculate band power ratios (useful for fatigue detection)
            verylow_power = band_powers["VeryLow"]

            for band_name in ["Low", "Mid", "High", "VeryHigh"]:
                band_power = band_powers[band_name]
                if verylow_power > 1e-10:
                    band_ratio = band_power / verylow_power
                else:
                    band_ratio = np.nan

                features[f"{prefix}STFT_{band_name}ToVeryLowRatio"] = float(band_ratio)
        except:
            for band_name in ["VeryLow", "Low", "Mid", "High", "VeryHigh"]:
                features[f"{prefix}STFT_{band_name}Power"] = np.nan

            for band_name in ["Low", "Mid", "High", "VeryHigh"]:
                features[f"{prefix}STFT_{band_name}ToVeryLowRatio"] = np.nan

        # 16) Time-varying fatigue analysis
        try:
            # Divide the signal into early, middle, and late segments
            if power_spectrum.shape[1] >= 3:
                n_windows = power_spectrum.shape[1]
                early_idx = max(1, n_windows // 3)
                late_idx = max(early_idx + 1, 2 * (n_windows // 3))

                # Calculate MNF for each segment
                early_ps = np.mean(power_spectrum[:, :early_idx], axis=1)
                middle_ps = np.mean(power_spectrum[:, early_idx:late_idx], axis=1)
                late_ps = np.mean(power_spectrum[:, late_idx:], axis=1)

                # Safe calculation of segment MNFs
                early_sum = np.sum(early_ps)
                middle_sum = np.sum(middle_ps)
                late_sum = np.sum(late_ps)

                early_mnf = np.sum(f * early_ps) / early_sum if early_sum > 1e-10 else np.nan
                middle_mnf = np.sum(f * middle_ps) / middle_sum if middle_sum > 1e-10 else np.nan
                late_mnf = np.sum(f * late_ps) / late_sum if late_sum > 1e-10 else np.nan

                # Store segment MNFs
                features[f"{prefix}STFT_EarlyMNF"] = float(early_mnf)
                features[f"{prefix}STFT_MiddleMNF"] = float(middle_mnf)
                features[f"{prefix}STFT_LateMNF"] = float(late_mnf)

                # Calculate fatigue indices based on segment comparisons
                if not np.isnan(early_mnf) and not np.isnan(late_mnf) and early_mnf > 1e-10:
                    mnf_fatigue_index = (early_mnf - late_mnf) / early_mnf  # Normalized decrease
                else:
                    mnf_fatigue_index = np.nan

                features[f"{prefix}STFT_MNF_FatigueIndex"] = float(mnf_fatigue_index)
            else:
                features[f"{prefix}STFT_EarlyMNF"] = np.nan
                features[f"{prefix}STFT_MiddleMNF"] = np.nan
                features[f"{prefix}STFT_LateMNF"] = np.nan
                features[f"{prefix}STFT_MNF_FatigueIndex"] = np.nan
        except:
            features[f"{prefix}STFT_EarlyMNF"] = np.nan
            features[f"{prefix}STFT_MiddleMNF"] = np.nan
            features[f"{prefix}STFT_LateMNF"] = np.nan
            features[f"{prefix}STFT_MNF_FatigueIndex"] = np.nan

    except Exception:
        # If STFT computation fails completely, return all NaN
        placeholders = [
            "STFT_MNF", "STFT_MDF", "STFT_PeakFreq", "STFT_TotalPower",
            "STFT_SpectralEntropy", "STFT_LowFreqPower", "STFT_HighFreqPower",
            "STFT_FreqBandRatio", "STFT_SpectralSlope", "STFT_MNF_Std",
            "STFT_MDF_Std", "STFT_Skewness", "STFT_DimitrovIndex",
            "STFT_SpectralMoment2", "STFT_SpectralMoment3", "STFT_MNF_Slope",
            "STFT_MNF_Decrease", "STFT_Bandwidth", "STFT_PeakFreqVariability",
            "STFT_VeryLowPower", "STFT_LowPower", "STFT_MidPower",
            "STFT_HighPower", "STFT_VeryHighPower", "STFT_LowToVeryLowRatio",
            "STFT_MidToVeryLowRatio", "STFT_HighToVeryLowRatio", "STFT_VeryHighToVeryLowRatio",
            "STFT_EarlyMNF", "STFT_MiddleMNF", "STFT_LateMNF", "STFT_MNF_FatigueIndex"
        ]
        for pl in placeholders:
            features[f"{prefix}{pl}"] = np.nan

    return features


def compute_emg_stft_features_multi_channel(
        emg_data: np.ndarray,
        sampling_rate: float,
        muscle_names: Optional[list] = None,
        window_size: int = 256,
        overlap: int = 128
) -> Dict[str, float]:
    """
    Compute STFT features for multi-channel EMG data.

    Parameters:
    -----------
    emg_data : np.ndarray
        EMG data shaped (n_channels, n_samples)
    sampling_rate : float
        Sampling rate of the EMG data in Hz
    muscle_names : list, optional
        List of muscle names for each channel
    window_size : int, default 256
        Size of the STFT window
    overlap : int, default 128
        Overlap between consecutive windows

    Returns:
    --------
    Dict[str, float]
        Dictionary of STFT features for all channels
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

        # Compute STFT features for this channel
        channel_features = compute_stft_features_for_signal(
            channel_data, sampling_rate, muscle_name, window_size, overlap
        )
        features.update(channel_features)

    # Compute features for the mean across all channels (if multiple channels)
    if n_channels > 1:
        mean_emg = np.mean(emg_data, axis=0)
        mean_features = compute_stft_features_for_signal(
            mean_emg, sampling_rate, "mean", window_size, overlap
        )
        features.update(mean_features)

    return features


def compute_emg_stft_features_for_window(
        window: Dict,
        fs_emg: float,
        muscle_names: Optional[list] = None,
        window_size: int = 256,
        overlap: int = 128
) -> Dict[str, float]:
    """
    Convenience function to extract STFT EMG features from a window dict.

    Parameters:
    -----------
    window : Dict
        Window dictionary containing 'emg' key with shape (n_channels, n_samples)
    fs_emg : float
        EMG sampling frequency
    muscle_names : list, optional
        List of muscle names for each channel
    window_size : int, default 256
        Size of the STFT window
    overlap : int, default 128
        Overlap between consecutive windows

    Returns:
    --------
    Dict[str, float]
        Dictionary of STFT EMG features
    """
    if "emg" not in window or window["emg"] is None:
        return {}

    emg_data = np.asarray(window["emg"])

    # Handle single channel case
    if emg_data.ndim == 1:
        emg_data = emg_data[np.newaxis, :]  # Add channel dimension

    return compute_emg_stft_features_multi_channel(
        emg_data, fs_emg, muscle_names, window_size, overlap
    )


def compute_emg_stft_features_per_repetition(
        emg_data: np.ndarray,
        repetition_segments: list,
        sampling_rate: float,
        muscle_names: Optional[list] = None,
        window_size: int = 256,
        overlap: int = 128
) -> Dict[int, Dict[str, float]]:
    """
    Compute STFT EMG features for each repetition segment.

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
    window_size : int, default 256
        Size of the STFT window
    overlap : int, default 128
        Overlap between consecutive windows

    Returns:
    --------
    Dict[int, Dict[str, float]]
        Dictionary mapping repetition number to STFT features
    """
    repetition_features = {}

    for rep_idx, (start_idx, end_idx) in enumerate(repetition_segments):
        # Extract data for this repetition
        rep_data = emg_data[:, start_idx:end_idx]

        if rep_data.shape[1] > window_size:  # Need enough samples for STFT
            # Compute STFT features for this repetition
            rep_features = compute_emg_stft_features_multi_channel(
                rep_data, sampling_rate, muscle_names, window_size, overlap
            )

            # Add repetition metadata
            rep_features["rep_duration"] = (end_idx - start_idx) / sampling_rate
            rep_features["rep_start_time"] = start_idx / sampling_rate
            rep_features["rep_end_time"] = end_idx / sampling_rate
            rep_features["rep_sample_count"] = end_idx - start_idx

            repetition_features[rep_idx + 1] = rep_features
        else:
            # Not enough data for STFT analysis
            repetition_features[rep_idx + 1] = {}

    return repetition_features

