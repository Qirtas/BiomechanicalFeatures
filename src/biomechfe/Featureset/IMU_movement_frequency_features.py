"""
Movement frequency feature extraction from IMU signals for biomechanical analysis.
Includes comprehensive frequency domain analysis, fatigue detection features,
tremor analysis, and time-frequency characteristics.
"""

import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks
from typing import Dict, Optional, Tuple
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

try:
    import pywt

    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False


def compute_frequency_spectrum(signal: np.ndarray, sampling_rate: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the frequency spectrum of a signal using FFT.

    Parameters:
    ----------
    signal : np.ndarray
        1D signal to analyze
    sampling_rate : float
        Sampling rate in Hz

    Returns:
    -------
    Tuple[np.ndarray, np.ndarray]
        (freqs, spectrum) - frequency bins and power at each bin
    """
    try:
        # Input validation
        if len(signal) < 2:
            return np.array([0]), np.array([0])

        # Remove any NaN values
        if np.isnan(signal).any():
            signal = np.nan_to_num(signal, nan=0.0)

        # Apply window to reduce spectral leakage
        window = np.hanning(len(signal))
        windowed_signal = signal * window

        # Compute FFT
        n = len(windowed_signal)
        yf = rfft(windowed_signal)
        xf = rfftfreq(n, 1 / sampling_rate)

        # Compute power spectrum (magnitude squared)
        power_spectrum = np.abs(yf) ** 2 / n

        return xf, power_spectrum

    except Exception:
        # Return minimal valid output
        return np.array([0]), np.array([0])


def compute_statistical_freq_features(freqs: np.ndarray, spectrum: np.ndarray, prefix: str) -> Dict[str, float]:
    """
    Statistical features of the frequency 'distribution'.

    Parameters:
    ----------
    freqs : np.ndarray
        Array of frequency bins
    spectrum : np.ndarray
        Power at each bin
    prefix : str
        String prefix for naming

    Returns:
    -------
    Dict[str, float]
        Dictionary of statistical frequency features
    """
    features = {}

    if len(freqs) < 2 or np.sum(spectrum) == 0:
        feature_names = ["MeanFrequency", "MedianFrequency", "FrequencyVariance",
                         "FrequencyRange", "FrequencyStd", "FrequencySkewness", "FrequencyKurtosis"]
        for f in feature_names:
            features[f"{prefix}_{f}"] = np.nan
        return features

    total_power = np.sum(spectrum)

    # 1. Mean Frequency
    mean_freq = np.sum(freqs * spectrum) / total_power

    # 2. Median Frequency
    half_power = total_power / 2.0
    cumsum_power = np.cumsum(spectrum)
    median_idx = np.searchsorted(cumsum_power, half_power)
    if median_idx >= len(freqs):
        median_idx = len(freqs) - 1
    median_freq = freqs[median_idx]

    # 3. Frequency Variance
    var_freq = np.sum(spectrum * (freqs - mean_freq) ** 2) / total_power

    # 4. Frequency Range
    valid_mask = (spectrum > 1e-12)
    if np.any(valid_mask):
        freq_min = freqs[valid_mask].min()
        freq_max = freqs[valid_mask].max()
        freq_range = freq_max - freq_min
    else:
        freq_range = 0.0

    freq_std = np.sqrt(var_freq)

    # Weighted moments for skew/kurt
    p = spectrum / total_power
    m3 = np.sum(p * (freqs - mean_freq) ** 3)
    m4 = np.sum(p * (freqs - mean_freq) ** 4)
    if freq_std > 1e-12:
        freq_skew = m3 / (freq_std ** 3)
        freq_kurt = (m4 / (freq_std ** 4)) - 3.0
    else:
        freq_skew = 0.0
        freq_kurt = -3.0

    features[f"{prefix}_MeanFrequency"] = mean_freq
    features[f"{prefix}_MedianFrequency"] = median_freq
    features[f"{prefix}_FrequencyVariance"] = var_freq
    features[f"{prefix}_FrequencyRange"] = freq_range
    features[f"{prefix}_FrequencyStd"] = freq_std
    features[f"{prefix}_FrequencySkewness"] = freq_skew
    features[f"{prefix}_FrequencyKurtosis"] = freq_kurt

    return features


def compute_frequency_domain_features(freqs: np.ndarray, spectrum: np.ndarray, prefix: str) -> Dict[str, float]:
    """
    Additional frequency-domain features.

    Parameters:
    ----------
    freqs : np.ndarray
        Array of frequency bins
    spectrum : np.ndarray
        Power at each bin
    prefix : str
        String prefix for naming

    Returns:
    -------
    Dict[str, float]
        Dictionary of frequency domain features
    """
    features = {}

    if len(freqs) < 2 or np.sum(spectrum) == 0:
        feature_names = ["PeakFrequency", "Bandwidth", "SpectralEntropy", "HarmonicEnergyRatio"]
        for f in feature_names:
            features[f"{prefix}_{f}"] = np.nan
        for band in ["Low", "Mid", "High"]:
            features[f"{prefix}_Energy_{band}"] = np.nan
        return features

    total_power = np.sum(spectrum)

    # 1. Peak Frequency
    peak_idx = np.argmax(spectrum)
    peak_freq = freqs[peak_idx]
    features[f"{prefix}_PeakFrequency"] = peak_freq

    # 2. Bandwidth (90% power)
    cum_power = np.cumsum(spectrum)
    power_90 = 0.90 * total_power
    start_idx = np.searchsorted(cum_power, (total_power - power_90))
    end_idx = np.searchsorted(cum_power, power_90)
    if start_idx >= len(freqs):
        start_idx = len(freqs) - 1
    if end_idx >= len(freqs):
        end_idx = len(freqs) - 1
    bandwidth = freqs[end_idx] - freqs[start_idx] if end_idx > start_idx else 0.0
    features[f"{prefix}_Bandwidth"] = bandwidth

    # 3. Spectral Entropy
    p = spectrum / total_power
    spectral_entropy = -np.sum(p * np.log2(p + 1e-12))
    features[f"{prefix}_SpectralEntropy"] = spectral_entropy

    # 4. Harmonic Energy Ratio
    harmonic_ratio = 0.0
    if peak_freq > 1e-6:
        tolerance = 0.02 * peak_freq
        max_harmonic = 5
        for k in range(max_harmonic):
            hf = peak_freq * (k + 1)
            idxs = np.where(np.abs(freqs - hf) <= tolerance)[0]
            harmonic_ratio += np.sum(spectrum[idxs])
        harmonic_ratio /= total_power
    features[f"{prefix}_HarmonicEnergyRatio"] = harmonic_ratio

    # 5. Energy in Low/Mid/High frequency bands
    low_mask = (freqs >= 0) & (freqs < 3)
    mid_mask = (freqs >= 3) & (freqs < 10)
    high_mask = (freqs >= 10)
    features[f"{prefix}_Energy_Low"] = np.sum(spectrum[low_mask])
    features[f"{prefix}_Energy_Mid"] = np.sum(spectrum[mid_mask])
    features[f"{prefix}_Energy_High"] = np.sum(spectrum[high_mask])

    return features


def compute_cycle_features(signal: np.ndarray, sampling_rate: float, prefix: str) -> Dict[str, float]:
    """
    Cycle & zero-crossing features.

    Parameters:
    ----------
    signal : np.ndarray
        1D signal to analyze
    sampling_rate : float
        Sampling rate in Hz
    prefix : str
        String prefix for naming

    Returns:
    -------
    Dict[str, float]
        Dictionary of cycle features
    """
    features = {}
    n = len(signal)

    if n < 2:
        features[f"{prefix}_CycleCount"] = np.nan
        features[f"{prefix}_ZeroCrossingRate"] = np.nan
        features[f"{prefix}_CycleFrequency"] = np.nan
        features[f"{prefix}_CycleDuration"] = np.nan
        return features

    signs = np.sign(signal)
    zc = np.sum(np.diff(signs) != 0)
    zero_cross_rate = zc / float(n)
    features[f"{prefix}_ZeroCrossingRate"] = zero_cross_rate

    cycle_count = zc / 2.0
    features[f"{prefix}_CycleCount"] = cycle_count

    total_time = n / float(sampling_rate)
    if total_time > 0:
        cycle_freq = cycle_count / total_time
    else:
        cycle_freq = np.nan
    features[f"{prefix}_CycleFrequency"] = cycle_freq

    if cycle_freq > 1e-12:
        cycle_duration = 1.0 / cycle_freq
    else:
        cycle_duration = np.nan
    features[f"{prefix}_CycleDuration"] = cycle_duration

    return features


def compute_time_frequency_features(signal: np.ndarray, sampling_rate: float, prefix: str) -> Dict[str, float]:
    """
    Time-Frequency features using short-time analysis.

    Parameters:
    ----------
    signal : np.ndarray
        1D signal to analyze
    sampling_rate : float
        Sampling rate in Hz
    prefix : str
        String prefix for naming

    Returns:
    -------
    Dict[str, float]
        Dictionary of time-frequency features
    """
    features = {}
    n = len(signal)

    if n < 2:
        features[f"{prefix}_InstantaneousFrequencyMean"] = np.nan
        features[f"{prefix}_FrequencyDrift"] = np.nan
        return features

    # Short-time approach
    window_size = int(0.5 * sampling_rate)
    step_size = window_size // 2
    peak_freqs = []

    start = 0
    while start < n:
        end = min(start + window_size, n)
        segment = signal[start:end]
        freqs, spec = compute_frequency_spectrum(segment, sampling_rate)
        if len(freqs) > 1:
            idx_peak = np.argmax(spec)
            pf = freqs[idx_peak]
        else:
            pf = 0.0
        peak_freqs.append(pf)
        start += step_size

    if len(peak_freqs) == 0:
        features[f"{prefix}_InstantaneousFrequencyMean"] = np.nan
        features[f"{prefix}_FrequencyDrift"] = np.nan
        return features

    inst_freq_mean = np.mean(peak_freqs)
    features[f"{prefix}_InstantaneousFrequencyMean"] = inst_freq_mean

    times = np.arange(len(peak_freqs)) * (step_size / float(sampling_rate))
    if len(times) > 1:
        b = np.polyfit(times, peak_freqs, 1)
        freq_drift = b[0]
    else:
        freq_drift = 0.0
    features[f"{prefix}_FrequencyDrift"] = freq_drift

    return features


def compute_fatigue_frequency_features(signal: np.ndarray, sampling_rate: float, prefix: str) -> Dict[str, float]:
    """
    Extract frequency features specifically designed for fatigue detection.

    Parameters:
    ----------
    signal : np.ndarray
        1D signal to analyze
    sampling_rate : float
        Sampling rate in Hz
    prefix : str
        String prefix for naming

    Returns:
    -------
    Dict[str, float]
        Dictionary of fatigue frequency features
    """
    features = {}

    try:
        # Input validation
        if len(signal) < 10:
            raise ValueError("Signal too short for fatigue frequency analysis")

        # Split signal into first and second half
        half_idx = len(signal) // 2
        first_half = signal[:half_idx]
        second_half = signal[half_idx:]

        if len(first_half) < 5 or len(second_half) < 5:
            raise ValueError("Signal halves too short for analysis")

        # Calculate frequency spectrum for each half
        freqs1, spec1 = compute_frequency_spectrum(first_half, sampling_rate)
        freqs2, spec2 = compute_frequency_spectrum(second_half, sampling_rate)

        # 1. Frequency stability: correlation between first and second half spectra
        min_len = min(len(spec1), len(spec2))
        if min_len > 1:
            spec1_interp = spec1[:min_len]
            spec2_interp = spec2[:min_len]
            if np.sum(spec1_interp) > 0 and np.sum(spec2_interp) > 0:
                freq_stability = np.corrcoef(spec1_interp, spec2_interp)[0, 1]
            else:
                freq_stability = 0.0
        else:
            freq_stability = 0.0

        # 2. Tremor index: ratio of high frequency (8-12 Hz) energy to total energy
        freqs_full, spec_full = compute_frequency_spectrum(signal, sampling_rate)
        tremor_mask = (freqs_full >= 8) & (freqs_full <= 12)

        if not np.any(tremor_mask):
            tremor_energy = 0
        else:
            tremor_energy = np.sum(spec_full[tremor_mask])

        total_energy = np.sum(spec_full)
        tremor_index = tremor_energy / total_energy if total_energy > 0 else 0

        # 3. Frequency fatigue: shift in median frequency from first to second half
        freq_fatigue = 0.0
        if len(freqs1) > 1 and len(freqs2) > 1:
            # Calculate median frequencies
            total_power1 = np.sum(spec1)
            total_power2 = np.sum(spec2)

            if total_power1 > 0 and total_power2 > 0:
                half_power1 = total_power1 / 2.0
                half_power2 = total_power2 / 2.0

                cumsum_power1 = np.cumsum(spec1)
                cumsum_power2 = np.cumsum(spec2)

                median_idx1 = np.searchsorted(cumsum_power1, half_power1)
                median_idx2 = np.searchsorted(cumsum_power2, half_power2)

                if median_idx1 >= len(freqs1):
                    median_idx1 = len(freqs1) - 1
                if median_idx2 >= len(freqs2):
                    median_idx2 = len(freqs2) - 1

                median_freq1 = freqs1[median_idx1]
                median_freq2 = freqs2[median_idx2]

                # Frequency fatigue is the relative shift in median frequency
                freq_fatigue = (median_freq1 - median_freq2) / median_freq1 if median_freq1 > 0 else 0

        features[f"{prefix}_FrequencyStability"] = freq_stability
        features[f"{prefix}_TremorIndex"] = tremor_index
        features[f"{prefix}_FrequencyFatigue"] = freq_fatigue

    except Exception:
        # Fill with NaN values
        features[f"{prefix}_FrequencyStability"] = np.nan
        features[f"{prefix}_TremorIndex"] = np.nan
        features[f"{prefix}_FrequencyFatigue"] = np.nan

    return features


def compute_wavelet_features(signal: np.ndarray, sampling_rate: float, prefix: str) -> Dict[str, float]:
    """
    Extract wavelet-based features that are robust to different signal types.

    Parameters:
    ----------
    signal : np.ndarray
        1D signal to analyze
    sampling_rate : float
        Sampling rate in Hz
    prefix : str
        String prefix for naming

    Returns:
    -------
    Dict[str, float]
        Dictionary of wavelet features
    """
    features = {}

    try:
        # Check for PyWavelets
        if not PYWT_AVAILABLE:
            # Fallback to FFT-based frequency bands
            return _compute_fft_based_wavelet_features(signal, sampling_rate, prefix)

        # Input validation and cleaning
        if len(signal) < 8:  # Need at least 8 points for level 3 decomposition
            feature_names = ["WaveletEnergy_Low", "WaveletEnergy_Mid", "WaveletEnergy_High", "WaveletTimeVariability"]
            for name in feature_names:
                features[f"{prefix}_{name}"] = np.nan
            return features

        # Clean signal
        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

        # Use discrete wavelet transform
        wavelet = 'db4'

        # Perform multilevel decomposition
        coeffs = pywt.wavedec(signal, wavelet, level=3)

        # Extract approximation and detail coefficients
        cA3, cD3, cD2, cD1 = coeffs

        # Calculate energy in each band
        energy_high = np.sum(cD1 ** 2)
        energy_mid = np.sum(cD2 ** 2)
        energy_low_mid = np.sum(cD3 ** 2)
        energy_low = np.sum(cA3 ** 2)

        # Combine low and low-mid for consistency
        energy_low_combined = energy_low + energy_low_mid

        # Calculate time variability as std of the detail coefficients
        time_variability = np.mean([np.std(cD1), np.std(cD2), np.std(cD3)])

        features[f"{prefix}_WaveletEnergy_Low"] = energy_low_combined
        features[f"{prefix}_WaveletEnergy_Mid"] = energy_mid
        features[f"{prefix}_WaveletEnergy_High"] = energy_high
        features[f"{prefix}_WaveletTimeVariability"] = time_variability

    except Exception:
        # Try FFT-based fallback
        return _compute_fft_based_wavelet_features(signal, sampling_rate, prefix)

    return features


def _compute_fft_based_wavelet_features(signal: np.ndarray, sampling_rate: float, prefix: str) -> Dict[str, float]:
    """
    Fallback FFT-based frequency bands when wavelets are not available.
    """
    features = {}

    try:
        # Compute FFT
        yf = rfft(signal)
        xf = rfftfreq(len(signal), 1 / sampling_rate)

        # Calculate power
        power = np.abs(yf) ** 2

        # Define frequency bands
        low_mask = (xf >= 0) & (xf < 3)
        mid_mask = (xf >= 3) & (xf < 10)
        high_mask = (xf >= 10)

        # Calculate energy in each band
        energy_low = np.sum(power[low_mask]) if np.any(low_mask) else 0
        energy_mid = np.sum(power[mid_mask]) if np.any(mid_mask) else 0
        energy_high = np.sum(power[high_mask]) if np.any(high_mask) else 0

        # Calculate time variability (using windowed std)
        window_size = min(10, len(signal) // 2)
        if window_size > 1:
            windows = [signal[i:i + window_size] for i in range(0, len(signal) - window_size, window_size // 2)]
            if windows:
                time_variability = np.mean([np.std(w) for w in windows])
            else:
                time_variability = np.std(signal)
        else:
            time_variability = np.std(signal)

        features[f"{prefix}_WaveletEnergy_Low"] = energy_low
        features[f"{prefix}_WaveletEnergy_Mid"] = energy_mid
        features[f"{prefix}_WaveletEnergy_High"] = energy_high
        features[f"{prefix}_WaveletTimeVariability"] = time_variability

    except Exception:
        # Fill with NaN values
        feature_names = ["WaveletEnergy_Low", "WaveletEnergy_Mid", "WaveletEnergy_High", "WaveletTimeVariability"]
        for name in feature_names:
            features[f"{prefix}_{name}"] = np.nan

    return features


def compute_enhanced_frequency_bands(freqs: np.ndarray, spectrum: np.ndarray, prefix: str) -> Dict[str, float]:
    """
    More detailed frequency band analysis with physiologically relevant bands.

    Parameters:
    ----------
    freqs : np.ndarray
        Array of frequency bins
    spectrum : np.ndarray
        Power at each bin
    prefix : str
        String prefix for naming

    Returns:
    -------
    Dict[str, float]
        Dictionary of enhanced frequency band features
    """
    features = {}

    try:
        # Input validation
        if len(freqs) < 2:
            raise ValueError("Too few frequency bins")

        if np.sum(spectrum) == 0:
            raise ValueError("Zero energy in spectrum")

        # Define physiologically relevant frequency bands
        bands = {
            "VeryLow": (0, 1),  # Very slow movements/postural drift
            "Low": (1, 3),  # Slow voluntary movements
            "Medium": (3, 7),  # Normal movement frequencies
            "High": (7, 12),  # Fast movements and early tremor
            "VeryHigh": (12, 25)  # Tremor and vibration
        }

        total_energy = np.sum(spectrum)

        for band_name, (low, high) in bands.items():
            mask = (freqs >= low) & (freqs < high)
            if not np.any(mask):
                band_energy = 0.0
                relative_energy = 0.0
            else:
                band_energy = np.sum(spectrum[mask])
                relative_energy = band_energy / total_energy

            features[f"{prefix}_Energy_{band_name}"] = band_energy
            features[f"{prefix}_RelativeEnergy_{band_name}"] = relative_energy

    except Exception:
        # Fill with NaN values
        for band in ["VeryLow", "Low", "Medium", "High", "VeryHigh"]:
            features[f"{prefix}_Energy_{band}"] = np.nan
            features[f"{prefix}_RelativeEnergy_{band}"] = np.nan

    return features


def compute_movement_frequency_features(
        signal: np.ndarray,
        sampling_rate: float,
        prefix: str
) -> Dict[str, float]:
    """
    Master function that integrates all frequency-based feature calculations for a single 1D signal.

    Parameters:
    ----------
    signal : np.ndarray
        1D signal to analyze
    sampling_rate : float
        Sampling rate in Hz
    prefix : str
        String prefix for naming

    Returns:
    -------
    Dict[str, float]
        Dictionary of movement frequency features
    """
    features = {}

    try:
        # Input validation
        if len(signal) < 5:
            raise ValueError(f"Input array too short: {len(signal)} points")

        if not np.isfinite(signal).all():
            signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

        # Compute frequency spectrum
        freqs, spectrum = compute_frequency_spectrum(signal, sampling_rate)

        # 1) Statistical frequency distribution features
        features.update(compute_statistical_freq_features(freqs, spectrum, prefix))

        # 2) Additional frequency-domain features
        features.update(compute_frequency_domain_features(freqs, spectrum, prefix))

        # 3) Cycle & zero-crossing features
        features.update(compute_cycle_features(signal, sampling_rate, prefix))

        # 4) Time-frequency features
        features.update(compute_time_frequency_features(signal, sampling_rate, prefix))

        # 5) Enhanced frequency band analysis
        features.update(compute_enhanced_frequency_bands(freqs, spectrum, prefix))

        # 6) Wavelet-based features
        features.update(compute_wavelet_features(signal, sampling_rate, prefix))

        # 7) Fatigue-specific frequency features
        features.update(compute_fatigue_frequency_features(signal, sampling_rate, prefix))

        return features

    except Exception:
        # Return empty features rather than failing completely
        default_features = [
            "FrequencyStability", "TremorIndex", "FrequencyFatigue",
            "WaveletEnergy_Low", "WaveletEnergy_Mid", "WaveletEnergy_High",
            "WaveletTimeVariability", "MeanFrequency", "MedianFrequency"
        ]
        return {f"{prefix}_{feat}": np.nan for feat in default_features}


def compute_movement_frequency_features_3axis(
        imu_data: np.ndarray,
        sampling_rate: float,
        sensor_type: str,
        site_name: Optional[str] = None
) -> Dict[str, float]:
    """
    Compute movement frequency features for 3-axis IMU data (accelerometer or gyroscope).

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

    Returns:
    -------
    Dict[str, float]
        Dictionary of movement frequency features
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

        # Compute movement frequency features for this axis
        axis_features = compute_movement_frequency_features(data, sampling_rate, prefix)
        features.update(axis_features)

    # Compute magnitude features
    if all(imu_data[i, :].size > 0 for i in range(3)):
        magnitude = np.sqrt(np.sum(imu_data ** 2, axis=0))
        mag_prefix = f"Magnitude{site_suffix}_{sensor_type}"
        mag_features = compute_movement_frequency_features(magnitude, sampling_rate, mag_prefix)
        features.update(mag_features)

    return features


def compute_emg_movement_frequency_features_for_window(
        window: Dict,
        fs_emg: float,
        site_name: Optional[str] = None
) -> Dict[str, float]:
    """
    Convenience function to extract movement frequency features from EMG data in a window dict.

    Parameters:
    -----------
    window : Dict
        Window dictionary containing 'emg' key with shape (n_channels, n_samples)
    fs_emg : float
        EMG sampling frequency
    site_name : str, optional
        EMG site name for feature naming (e.g., 'biceps', 'deltoid')

    Returns:
    --------
    Dict[str, float]
        Dictionary of EMG movement frequency features
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

        # Compute movement frequency features for this EMG channel
        channel_features = compute_movement_frequency_features(
            channel_data, fs_emg, prefix
        )
        features.update(channel_features)

    # Also compute features for the mean across all channels (if multiple channels)
    if emg_data.shape[0] > 1:
        mean_emg = np.mean(emg_data, axis=0)
        if site_name:
            mean_prefix = f"mean_{site_name}_emg"
        else:
            mean_prefix = "mean_emg"

        mean_features = compute_movement_frequency_features(
            mean_emg, fs_emg, mean_prefix
        )
        features.update(mean_features)

    return features

def compute_movement_frequency_features_for_window(
        window: Dict,
        fs_imu: float,
        site_name: Optional[str] = None,
        include_acc: bool = True,
        include_gyr: bool = True
) -> Dict[str, float]:
    """
    Convenience function to extract movement frequency features from a window dict.

    Parameters:
    -----------
    window : Dict
        Window dictionary containing 'acc' and/or 'gyr' keys
    fs_imu : float
        IMU sampling frequency
    site_name : str, optional
        Sensor site name for feature naming
    include_acc : bool, default True
        Whether to include accelerometer frequency features
    include_gyr : bool, default True
        Whether to include gyroscope frequency features

    Returns:
    --------
    Dict[str, float]
        Dictionary of movement frequency features
    """
    features = {}

    # Process accelerometer data
    if include_acc and "acc" in window and window["acc"] is not None:
        acc_data = np.asarray(window["acc"])
        acc_features = compute_movement_frequency_features_3axis(
            acc_data, fs_imu, "acc", site_name
        )
        features.update(acc_features)

    # Process gyroscope data
    if include_gyr and "gyr" in window and window["gyr"] is not None:
        gyr_data = np.asarray(window["gyr"])
        gyr_features = compute_movement_frequency_features_3axis(
            gyr_data, fs_imu, "gyr", site_name
        )
        features.update(gyr_features)

    return features
