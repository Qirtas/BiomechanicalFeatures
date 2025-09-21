"""
Maximum Power Spectral Density (MPSD) feature extraction from IMU signals.
Includes comprehensive spectral analysis, fatigue indicators, and frequency domain
characteristics specifically designed for biomechanical movement analysis.
"""

import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.stats import skew, kurtosis
from typing import Dict, Optional, Tuple


def compute_psd(signal: np.ndarray, sampling_rate: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the Power Spectral Density (PSD) of a 1D signal using a real FFT.
    Applies windowing to reduce spectral leakage.

    Parameters:
    ----------
    signal : np.ndarray
        1D signal to analyze
    sampling_rate : float
        Sampling rate in Hz

    Returns:
    -------
    Tuple[np.ndarray, np.ndarray]
        (freqs, psd) - Array of frequency bins (0 .. Nyquist) and
        power spectral density values for each bin
    """
    n = len(signal)
    if n < 2:
        # Edge case: not enough samples
        return np.array([0.0]), np.array([0.0])

    # Apply Hanning window to reduce spectral leakage
    window = np.hanning(n)
    windowed_signal = signal * window

    # Compute scaling factor to correct for window energy loss
    # This ensures the total power is preserved
    scale = 1.0 / np.mean(window ** 2)

    # Real FFT
    fft_vals = rfft(windowed_signal)
    freqs = rfftfreq(n, d=1.0 / sampling_rate)

    # Power spectral density (magnitude squared) with window correction
    psd = np.abs(fft_vals) ** 2 * scale / n

    return freqs, psd


def compute_mpsd_features(
        freqs: np.ndarray,
        psd: np.ndarray,
        prefix: str,
        low_freq_threshold: float = 5.0
) -> Dict[str, float]:
    """
    Given frequency bins (freqs) and power spectral density (psd),
    compute MPSD-related features.

    Parameters:
    ----------
    freqs : np.ndarray
        Array of frequency bins
    psd : np.ndarray
        Power spectral density values
    prefix : str
        Label for the features (e.g., "X_Shoulder_acc")
    low_freq_threshold : float, default 5.0
        Frequency cut-off to separate low vs high freq power (Hz)

    Returns:
    -------
    Dict[str, float]
        Dictionary of MPSD features
    """
    features = {}
    n_bins = len(freqs)

    if n_bins < 2 or np.all(psd == 0):
        # Edge case: empty or no power
        placeholders = [
            "MPSD", "Freq_MPSD", "Power_Dominant_Band", "Ratio_Dominant_Band",
            "Total_LowFreq_Power", "Total_HighFreq_Power", "LowHigh_FreqRatio",
            "Total_Power", "PSD_Skewness", "PSD_Kurtosis", "DominantFreq_Range",
            "MedianFreq", "BandwidthHalfPower"
        ]
        for p in placeholders:
            features[f"{prefix}_{p}"] = np.nan
        return features

    # 1) Maximum Power Spectral Density (MPSD)
    idx_mpsd = np.argmax(psd)
    mpsd_val = psd[idx_mpsd]
    features[f"{prefix}_MPSD"] = float(mpsd_val)

    # 2) Frequency of MPSD
    freq_mpsd = freqs[idx_mpsd]
    features[f"{prefix}_Freq_MPSD"] = float(freq_mpsd)

    # 3) Power in Dominant Frequency Band (± 5% of freq_mpsd)
    band_power = 0.0
    if freq_mpsd > 1e-12:
        band_low = freq_mpsd * 0.95
        band_high = freq_mpsd * 1.05
        band_mask = (freqs >= band_low) & (freqs <= band_high)
        band_power = np.sum(psd[band_mask])
    features[f"{prefix}_Power_Dominant_Band"] = float(band_power)

    # 4) Ratio of Power in Dominant Frequency Band
    total_power = np.sum(psd)
    ratio_dominant_band = band_power / total_power if total_power > 1e-12 else 0.0
    features[f"{prefix}_Ratio_Dominant_Band"] = float(ratio_dominant_band)

    # 5) Total Low-Frequency Power (< low_freq_threshold)
    low_mask = (freqs < low_freq_threshold)
    low_freq_power = np.sum(psd[low_mask])
    features[f"{prefix}_Total_LowFreq_Power"] = float(low_freq_power)

    # 6) Total High-Frequency Power (>= low_freq_threshold)
    high_mask = (freqs >= low_freq_threshold)
    high_freq_power = np.sum(psd[high_mask])
    features[f"{prefix}_Total_HighFreq_Power"] = float(high_freq_power)

    # 7) Low-to-High Frequency Power Ratio
    if high_freq_power > 1e-12:
        low_high_ratio = low_freq_power / high_freq_power
    else:
        low_high_ratio = np.inf if low_freq_power > 0 else np.nan
    features[f"{prefix}_LowHigh_FreqRatio"] = float(low_high_ratio)

    # 8) Total Power (Energy)
    features[f"{prefix}_Total_Power"] = float(total_power)

    # 9) PSD Skewness
    psd_mean = np.mean(psd)
    psd_std = np.std(psd, ddof=1) if len(psd) > 1 else 0.0
    if psd_std > 1e-12:
        psd_skew = np.mean(((psd - psd_mean) / psd_std) ** 3)
        psd_kurt = np.mean(((psd - psd_mean) / psd_std) ** 4) - 3.0
    else:
        psd_skew = 0.0
        psd_kurt = -3.0

    features[f"{prefix}_PSD_Skewness"] = float(psd_skew)

    # 10) PSD Kurtosis
    features[f"{prefix}_PSD_Kurtosis"] = float(psd_kurt)

    # 11) Dominant Frequency Range
    try:
        cumsum_psd = np.cumsum(np.sort(psd))
        sorted_psd = np.sort(psd)
        total = cumsum_psd[-1]
        lower_thresh = 0.05 * total
        upper_thresh = 0.95 * total

        # Map sorted PSD back to frequencies
        psd_order = np.argsort(psd)  # indices that would sort PSD
        freq_sorted = freqs[psd_order]
        cumsum_psd_freq = np.cumsum(psd[psd_order])

        lower_idx = np.searchsorted(cumsum_psd_freq, lower_thresh)
        if lower_idx >= len(psd_order):
            lower_idx = len(psd_order) - 1

        upper_idx = np.searchsorted(cumsum_psd_freq, upper_thresh)
        if upper_idx >= len(psd_order):
            upper_idx = len(psd_order) - 1

        freq_range = abs(freq_sorted[upper_idx] - freq_sorted[lower_idx])
        features[f"{prefix}_DominantFreq_Range"] = float(freq_range)
    except Exception:
        features[f"{prefix}_DominantFreq_Range"] = np.nan

    # 12) Median Frequency
    # This is a key fatigue indicator as it typically shifts to lower frequencies with fatigue
    try:
        if total_power > 1e-12:
            # Sort frequencies and corresponding PSD values
            sorted_indices = np.argsort(freqs)
            sorted_freqs = freqs[sorted_indices]
            sorted_psd = psd[sorted_indices]

            # Calculate cumulative power
            cumulative_power = np.cumsum(sorted_psd)

            # Normalize to get percentage of total power
            cumulative_power_pct = cumulative_power / total_power

            # Find the frequency at which cumulative power reaches 50%
            median_idx = np.searchsorted(cumulative_power_pct, 0.5)
            if median_idx >= len(sorted_freqs):
                median_idx = len(sorted_freqs) - 1

            median_freq = sorted_freqs[median_idx]
        else:
            median_freq = 0.0
    except Exception:
        median_freq = 0.0

    features[f"{prefix}_MedianFreq"] = float(median_freq)

    # 13) Bandwidth at Half Maximum Power
    # This captures the spread of power around the dominant frequency
    # Fatigue often leads to changes in this bandwidth
    try:
        if mpsd_val > 1e-12:
            # Half maximum power threshold
            half_max_power = mpsd_val / 2.0

            # Find all frequencies where power exceeds half maximum
            half_power_mask = psd >= half_max_power

            if np.any(half_power_mask):
                # Get frequencies that exceed half maximum power
                half_power_freqs = freqs[half_power_mask]

                # Calculate bandwidth (max - min frequency)
                bandwidth = np.max(half_power_freqs) - np.min(half_power_freqs)
            else:
                # No frequencies exceed half maximum power
                bandwidth = 0.0
        else:
            bandwidth = 0.0
    except Exception:
        bandwidth = 0.0

    features[f"{prefix}_BandwidthHalfPower"] = float(bandwidth)

    return features


def compute_advanced_mpsd_features(
        freqs: np.ndarray,
        psd: np.ndarray,
        prefix: str,
        sampling_rate: float
) -> Dict[str, float]:
    """
    Compute advanced MPSD features including spectral centroid, rolloff, and fatigue indicators.

    Parameters:
    ----------
    freqs : np.ndarray
        Array of frequency bins
    psd : np.ndarray
        Power spectral density values
    prefix : str
        Label for the features
    sampling_rate : float
        Sampling rate in Hz

    Returns:
    -------
    Dict[str, float]
        Dictionary of advanced MPSD features
    """
    features = {}

    if len(freqs) < 2 or np.sum(psd) == 0:
        advanced_feature_names = [
            "SpectralCentroid", "SpectralRolloff", "SpectralSpread",
            "SpectralFlux", "SpectralFlatness", "SpectralCrest",
            "FrequencyStability", "PowerConcentration"
        ]
        for f in advanced_feature_names:
            features[f"{prefix}_{f}"] = np.nan
        return features

    total_power = np.sum(psd)

    # 1) Spectral Centroid (center of mass of the spectrum)
    try:
        spectral_centroid = np.sum(freqs * psd) / total_power
        features[f"{prefix}_SpectralCentroid"] = float(spectral_centroid)
    except Exception:
        features[f"{prefix}_SpectralCentroid"] = np.nan

    # 2) Spectral Rolloff (frequency below which 85% of energy is contained)
    try:
        cumulative_power = np.cumsum(psd)
        rolloff_threshold = 0.85 * total_power
        rolloff_idx = np.searchsorted(cumulative_power, rolloff_threshold)
        if rolloff_idx >= len(freqs):
            rolloff_idx = len(freqs) - 1
        spectral_rolloff = freqs[rolloff_idx]
        features[f"{prefix}_SpectralRolloff"] = float(spectral_rolloff)
    except Exception:
        features[f"{prefix}_SpectralRolloff"] = np.nan

    # 3) Spectral Spread (variance around spectral centroid)
    try:
        spectral_centroid = np.sum(freqs * psd) / total_power
        spectral_spread = np.sqrt(np.sum(psd * (freqs - spectral_centroid) ** 2) / total_power)
        features[f"{prefix}_SpectralSpread"] = float(spectral_spread)
    except Exception:
        features[f"{prefix}_SpectralSpread"] = np.nan

    # 4) Spectral Flux (rate of change in spectrum - requires temporal context)
    # For single window, we approximate using high-frequency content
    try:
        high_freq_mask = freqs > (sampling_rate / 8)  # Upper frequency region
        high_freq_power = np.sum(psd[high_freq_mask])
        spectral_flux = high_freq_power / total_power
        features[f"{prefix}_SpectralFlux"] = float(spectral_flux)
    except Exception:
        features[f"{prefix}_SpectralFlux"] = np.nan

    # 5) Spectral Flatness (Wiener entropy - measure of noisiness)
    try:
        # Geometric mean / Arithmetic mean
        psd_nonzero = psd[psd > 1e-12]  # Avoid log(0)
        if len(psd_nonzero) > 0:
            geometric_mean = np.exp(np.mean(np.log(psd_nonzero)))
            arithmetic_mean = np.mean(psd_nonzero)
            spectral_flatness = geometric_mean / arithmetic_mean
        else:
            spectral_flatness = 0.0
        features[f"{prefix}_SpectralFlatness"] = float(spectral_flatness)
    except Exception:
        features[f"{prefix}_SpectralFlatness"] = np.nan

    # 6) Spectral Crest Factor (peak-to-average ratio)
    try:
        spectral_crest = np.max(psd) / np.mean(psd) if np.mean(psd) > 0 else 0.0
        features[f"{prefix}_SpectralCrest"] = float(spectral_crest)
    except Exception:
        features[f"{prefix}_SpectralCrest"] = np.nan

    # 7) Frequency Stability (inverse of spectral spread, normalized)
    try:
        spectral_centroid = np.sum(freqs * psd) / total_power
        spectral_spread = np.sqrt(np.sum(psd * (freqs - spectral_centroid) ** 2) / total_power)
        frequency_stability = 1.0 / (1.0 + spectral_spread) if spectral_spread >= 0 else 0.0
        features[f"{prefix}_FrequencyStability"] = float(frequency_stability)
    except Exception:
        features[f"{prefix}_FrequencyStability"] = np.nan

    # 8) Power Concentration (how concentrated the power is around dominant frequency)
    try:
        max_power = np.max(psd)
        # Define a narrow band around the peak
        peak_idx = np.argmax(psd)
        peak_freq = freqs[peak_idx]

        # ±10% band around peak frequency
        band_width = max(0.1 * peak_freq, freqs[1] - freqs[0])  # At least one bin
        band_mask = np.abs(freqs - peak_freq) <= band_width
        concentrated_power = np.sum(psd[band_mask])

        power_concentration = concentrated_power / total_power
        features[f"{prefix}_PowerConcentration"] = float(power_concentration)
    except Exception:
        features[f"{prefix}_PowerConcentration"] = np.nan

    return features


def compute_mpsd_for_signal(
        signal: np.ndarray,
        sampling_rate: float,
        prefix: str,
        low_freq_threshold: float = 5.0,
        include_advanced: bool = False
) -> Dict[str, float]:
    """
    Wrapper to compute MPSD features for a single 1D signal.

    Parameters:
    ----------
    signal : np.ndarray
        1D signal to analyze
    sampling_rate : float
        Sampling rate in Hz
    prefix : str
        Prefix for feature names
    low_freq_threshold : float, default 5.0
        Frequency threshold for low/high frequency separation
    include_advanced : bool, default False
        Whether to include advanced spectral features

    Returns:
    -------
    Dict[str, float]
        Dictionary of MPSD features
    """
    freqs, psd = compute_psd(signal, sampling_rate)
    features = compute_mpsd_features(freqs, psd, prefix, low_freq_threshold)

    if include_advanced:
        advanced_features = compute_advanced_mpsd_features(freqs, psd, prefix, sampling_rate)
        features.update(advanced_features)

    return features


def compute_mpsd_features_3axis(
        imu_data: np.ndarray,
        sampling_rate: float,
        sensor_type: str,
        site_name: Optional[str] = None,
        low_freq_threshold: float = 5.0,
        include_advanced: bool = False
) -> Dict[str, float]:
    """
    Compute MPSD features for 3-axis IMU data (accelerometer or gyroscope).

    Parameters:
    ----------
    imu_data : np.ndarray
        IMU data shaped (3, n_samples) where rows are [X, Y, Z]
    sampling_rate : float
        Sampling rate in Hz
    sensor_type : str
        Type of sensor ('acc' for accelerometer, 'gyr' for gyroscope)
    site_name : str, optional
        Name of the sensor site for feature naming
    low_freq_threshold : float, default 5.0
        Frequency threshold for low/high frequency separation
    include_advanced : bool, default False
        Whether to include advanced spectral features

    Returns:
    -------
    Dict[str, float]
        Dictionary of MPSD features
    """
    if imu_data.shape[0] != 3:
        raise ValueError(f"Expected imu_data shape (3, n_samples), got {imu_data.shape}")

    features = {}
    site_suffix = f"_{site_name}" if site_name else ""

    # Process each axis individually
    axis_names = ['X', 'Y', 'Z']
    for i, axis in enumerate(axis_names):
        data = imu_data[i, :]
        prefix = f"{axis}{site_suffix}_{sensor_type}"

        if data.size == 0:
            continue

        # Compute MPSD features for this axis
        axis_features = compute_mpsd_for_signal(
            data, sampling_rate, prefix, low_freq_threshold, include_advanced
        )
        features.update(axis_features)

    # Compute magnitude features
    if all(imu_data[i, :].size > 0 for i in range(3)):
        magnitude = np.sqrt(np.sum(imu_data ** 2, axis=0))
        mag_prefix = f"Magnitude{site_suffix}_{sensor_type}"
        mag_features = compute_mpsd_for_signal(
            magnitude, sampling_rate, mag_prefix, low_freq_threshold, include_advanced
        )
        features.update(mag_features)

    return features


def compute_mpsd_features_for_window(
        window: Dict,
        fs_imu: float,
        site_name: Optional[str] = None,
        low_freq_threshold: float = 5.0,
        include_advanced: bool = False,
        include_acc: bool = True,
        include_gyr: bool = True
) -> Dict[str, float]:
    """
    Convenience function to extract MPSD features from a window dict.

    Parameters:
    -----------
    window : Dict
        Window dictionary containing 'acc' and/or 'gyr' keys
    fs_imu : float
        IMU sampling frequency
    site_name : str, optional
        Sensor site name for feature naming
    low_freq_threshold : float, default 5.0
        Frequency threshold for low/high frequency separation
    include_advanced : bool, default False
        Whether to include advanced spectral features
    include_acc : bool, default True
        Whether to include accelerometer MPSD features
    include_gyr : bool, default True
        Whether to include gyroscope MPSD features

    Returns:
    --------
    Dict[str, float]
        Dictionary of MPSD features
    """
    features = {}

    # Process accelerometer data
    if include_acc and "acc" in window and window["acc"] is not None:
        acc_data = np.asarray(window["acc"])
        acc_features = compute_mpsd_features_3axis(
            acc_data, fs_imu, "acc", site_name, low_freq_threshold, include_advanced
        )
        features.update(acc_features)

    # Process gyroscope data
    if include_gyr and "gyr" in window and window["gyr"] is not None:
        gyr_data = np.asarray(window["gyr"])
        gyr_features = compute_mpsd_features_3axis(
            gyr_data, fs_imu, "gyr", site_name, low_freq_threshold, include_advanced
        )
        features.update(gyr_features)

    return features


def compute_emg_mpsd_features_for_window(
        window: Dict,
        fs_emg: float,
        site_name: Optional[str] = None,
        low_freq_threshold: float = 150.0,  # Different threshold for EMG
        include_advanced: bool = False
) -> Dict[str, float]:
    """
    Convenience function to extract MPSD features from EMG data in a window dict.

    Parameters:
    -----------
    window : Dict
        Window dictionary containing 'emg' key with shape (n_channels, n_samples)
    fs_emg : float
        EMG sampling frequency
    site_name : str, optional
        EMG site name for feature naming
    low_freq_threshold : float, default 150.0
        Frequency threshold for EMG low/high frequency separation
    include_advanced : bool, default False
        Whether to include advanced spectral features

    Returns:
    --------
    Dict[str, float]
        Dictionary of EMG MPSD features
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
            prefix = f"ch{ch_idx}_{site_name}_emg"
        else:
            prefix = f"ch{ch_idx}_emg"

        # Compute MPSD features for this EMG channel
        channel_features = compute_mpsd_for_signal(
            channel_data, fs_emg, prefix, low_freq_threshold, include_advanced
        )
        features.update(channel_features)

    # Also compute features for the mean across all channels (if multiple channels)
    if emg_data.shape[0] > 1:
        mean_emg = np.mean(emg_data, axis=0)
        if site_name:
            mean_prefix = f"mean_{site_name}_emg"
        else:
            mean_prefix = "mean_emg"

        mean_features = compute_mpsd_for_signal(
            mean_emg, fs_emg, mean_prefix, low_freq_threshold, include_advanced
        )
        features.update(mean_features)

    return features
