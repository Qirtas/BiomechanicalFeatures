"""
Comprehensive statistical feature extraction for EMG signals.
Includes basic statistical features, fatigue-specific indicators, frequency domain analysis,
signal complexity measures, and advanced EMG characteristics for biomechanical analysis.
"""

import numpy as np
from scipy.stats import skew, kurtosis, entropy
from typing import Dict, Optional

# Optional imports for advanced features
try:
    from scipy.signal import hilbert, find_peaks, butter, filtfilt, welch

    SCIPY_SIGNAL_AVAILABLE = True
except ImportError:
    SCIPY_SIGNAL_AVAILABLE = False


def compute_signal_energy(signal: np.ndarray) -> float:
    """
    Energy = sum(x[i]^2) over the entire signal.
    """
    return float(np.sum(signal ** 2))


def compute_mean_power(signal: np.ndarray) -> float:
    """
    Mean Power = mean(x[i]^2).
    """
    return float(np.mean(signal ** 2))


def compute_snr(signal: np.ndarray) -> float:
    """
    Signal-to-Noise Ratio (SNR) in dB.
    """
    signal_power = np.mean(signal ** 2)
    # Approximate noise as standard deviation around mean
    noise = np.std(signal - np.mean(signal))
    noise_power = noise ** 2 if noise != 0 else 1e-12
    snr_value = 10.0 * np.log10(signal_power / noise_power)
    return float(snr_value)


def compute_mean_frequency(signal: np.ndarray, fs: float = 1000) -> float:
    """
    Mean frequency of the signal.
    """
    n = len(signal)
    if n < 2:
        return np.nan

    # Real FFT
    fft_vals = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    # Power
    power = np.abs(fft_vals) ** 2
    total_power = np.sum(power)
    if total_power < 1e-12:
        return 0.0

    mean_freq = np.sum(freqs * power) / total_power
    return float(mean_freq)


def compute_median_frequency(signal: np.ndarray, fs: float = 1000) -> float:
    """
    Median frequency of the signal - a key fatigue indicator.
    """
    n = len(signal)
    if n < 4:
        return np.nan

    try:
        if SCIPY_SIGNAL_AVAILABLE:
            # Use Welch's method for more robust PSD estimation
            freqs, psd = welch(signal, fs=fs, nperseg=min(256, n // 2), noverlap=min(128, n // 4))
        else:
            # Fallback to simple FFT
            fft_vals = np.fft.rfft(signal)
            freqs = np.fft.rfftfreq(n, d=1.0 / fs)
            psd = np.abs(fft_vals) ** 2

        # Calculate cumulative sum of power
        cumulative_power = np.cumsum(psd)
        total_power = cumulative_power[-1]

        if total_power < 1e-12:
            return np.nan

        # Find frequency at which cumulative power reaches 50% of total power
        median_idx = np.argmax(cumulative_power >= 0.5 * total_power)
        median_freq = freqs[median_idx]

        return float(median_freq)
    except Exception:
        return np.nan


def compute_frequency_ratio(signal: np.ndarray, fs: float = 1000,
                            low_cutoff: float = 20, high_cutoff: float = 250) -> float:
    """
    Calculate the ratio of low frequency power to high frequency power.
    This ratio increases with fatigue.
    """
    n = len(signal)
    if n < 4:
        return np.nan

    try:
        if SCIPY_SIGNAL_AVAILABLE:
            freqs, psd = welch(signal, fs=fs, nperseg=min(256, n // 2), noverlap=min(128, n // 4))
        else:
            fft_vals = np.fft.rfft(signal)
            freqs = np.fft.rfftfreq(n, d=1.0 / fs)
            psd = np.abs(fft_vals) ** 2

        # Define frequency bands
        low_mask = (freqs <= low_cutoff) & (freqs > 0)  # Exclude DC component
        high_mask = (freqs >= high_cutoff)

        # Calculate power in each band
        low_power = np.sum(psd[low_mask]) if np.any(low_mask) else 0
        high_power = np.sum(psd[high_mask]) if np.any(high_mask) else 0

        # Calculate ratio (avoid division by zero)
        if high_power < 1e-12:
            return np.nan

        return float(low_power / high_power)
    except Exception:
        return np.nan


def compute_muscle_fatigue_index(signal: np.ndarray, fs: float = 1000) -> float:
    """
    Calculate Muscle Fatigue Index (MFI) based on spectral changes.
    MFI increases with fatigue.
    """
    n = len(signal)
    if n < 4:
        return np.nan

    try:
        # Calculate median frequency
        median_freq = compute_median_frequency(signal, fs)

        # Calculate mean frequency
        mean_freq = compute_mean_frequency(signal, fs)

        if np.isnan(median_freq) or np.isnan(mean_freq) or mean_freq < 1e-12:
            return np.nan

        # MFI is the ratio of median to mean frequency
        # This ratio decreases with fatigue
        mfi = median_freq / mean_freq

        # Invert so that higher values indicate more fatigue
        return float(1.0 / mfi) if mfi > 0 else np.nan
    except Exception:
        return np.nan


def compute_sample_entropy(signal: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """
    Calculate sample entropy of the signal.
    Lower values indicate more regular/predictable patterns.
    """
    if len(signal) < m + 2:
        return np.nan

    try:
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
        vectors_m = create_vectors(signal, m)
        vectors_m1 = create_vectors(signal, m + 1)

        B_m = count_matches(vectors_m, r)
        B_m1 = count_matches(vectors_m1, r)

        # Avoid log(0)
        if B_m == 0 or B_m1 == 0:
            return np.nan

        return float(-np.log(B_m1 / B_m))
    except Exception:
        return np.nan


def estimate_conduction_velocity(signal: np.ndarray, fs: float = 1000,
                                 electrode_distance: float = 10) -> float:
    """
    Estimate muscle fiber conduction velocity.
    Conduction velocity decreases with fatigue.
    """
    if len(signal) < 10:
        return np.nan

    try:
        # Simplified approach using autocorrelation to find the delay
        n = len(signal)
        signal_centered = signal - np.mean(signal)
        autocorr = np.correlate(signal_centered, signal_centered, mode='full')[n - 1:]

        # Find the first peak after lag 0
        if SCIPY_SIGNAL_AVAILABLE:
            peaks, _ = find_peaks(autocorr, height=0)
        else:
            # Simple peak finding fallback
            peaks = []
            for i in range(1, len(autocorr) - 1):
                if autocorr[i] > autocorr[i - 1] and autocorr[i] > autocorr[i + 1] and autocorr[i] > 0:
                    peaks.append(i)
            peaks = np.array(peaks)

        if len(peaks) == 0 or peaks[0] == 0:
            return np.nan

        # Time delay in seconds
        time_delay = peaks[0] / fs

        if time_delay < 1e-6:  # Avoid division by very small numbers
            return np.nan

        # Convert mm to m and calculate velocity
        distance_m = electrode_distance / 1000.0
        velocity = distance_m / time_delay  # m/s

        # Typical range for muscle fiber conduction velocity is 2-6 m/s
        if velocity < 1 or velocity > 10:
            return np.nan

        return float(velocity)
    except Exception:
        return np.nan


def bandpass_filter_emg(signal: np.ndarray, fs: float = 1000,
                        lowcut: float = 20, highcut: float = 450, order: int = 4) -> np.ndarray:
    """
    Apply bandpass filter to EMG signal with robust error handling.
    """
    # Check input signal
    if len(signal) < order * 3:
        return signal

    # Check for NaN or Inf values
    if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
        signal = np.nan_to_num(signal)

    # Check sampling rate
    if fs <= 0:
        fs = 1000

    # Validate frequency parameters
    if lowcut <= 0:
        lowcut = 1.0

    if highcut >= fs / 2:
        highcut = fs / 2 - 1

    if lowcut >= highcut:
        return signal

    # Calculate normalized frequencies
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    # Final validation of normalized frequencies
    if low <= 0 or high >= 1:
        return signal

    if not SCIPY_SIGNAL_AVAILABLE:
        return signal

    try:
        # Design and apply filter
        b, a = butter(order, [low, high], btype='band')
        filtered_signal = filtfilt(b, a, signal)

        # Check if filtering produced valid results
        if np.any(np.isnan(filtered_signal)) or np.any(np.isinf(filtered_signal)):
            return signal

        return filtered_signal
    except Exception:
        return signal


def extract_muap_features(signal: np.ndarray, fs: float = 1000) -> Dict[str, float]:
    """
    Extract Motor Unit Action Potential (MUAP) features.
    MUAP characteristics change with fatigue.
    """
    features = {}

    if len(signal) < 20:
        features["MUAP_Count"] = np.nan
        features["MUAP_MeanAmplitude"] = np.nan
        features["MUAP_MeanDuration"] = np.nan
        features["MUAP_MeanArea"] = np.nan
        features["MUAP_FiringRate"] = np.nan
        return features

    try:
        # Preprocess signal
        signal_centered = signal - np.mean(signal)

        # Detect peaks (potential MUAPs)
        height_threshold = 1.5 * np.std(signal_centered)
        min_distance = max(1, int(0.005 * fs))  # Minimum 5ms between peaks

        if SCIPY_SIGNAL_AVAILABLE:
            peaks, properties = find_peaks(np.abs(signal_centered),
                                           height=height_threshold,
                                           distance=min_distance)
        else:
            # Simple peak finding fallback
            abs_signal = np.abs(signal_centered)
            peaks = []
            for i in range(min_distance, len(abs_signal) - min_distance):
                if (abs_signal[i] > height_threshold and
                        abs_signal[i] > abs_signal[i - 1] and
                        abs_signal[i] > abs_signal[i + 1]):
                    # Check minimum distance
                    if not peaks or i - peaks[-1] >= min_distance:
                        peaks.append(i)
            peaks = np.array(peaks)
            properties = {"peak_heights": abs_signal[peaks] if len(peaks) > 0 else np.array([])}

        if len(peaks) == 0:
            features["MUAP_Count"] = 0
            features["MUAP_MeanAmplitude"] = np.nan
            features["MUAP_MeanDuration"] = np.nan
            features["MUAP_MeanArea"] = np.nan
            features["MUAP_FiringRate"] = np.nan
            return features

        # Count of detected MUAPs
        features["MUAP_Count"] = len(peaks)

        # Mean amplitude of MUAPs
        amplitudes = properties["peak_heights"]
        features["MUAP_MeanAmplitude"] = np.mean(amplitudes)

        # Estimate MUAP durations (simplified)
        durations = []
        areas = []

        for peak in peaks:
            # Find start and end of MUAP (simplified)
            start = max(0, peak - int(0.01 * fs))  # Look 10ms before peak
            end = min(len(signal_centered), peak + int(0.01 * fs))  # Look 10ms after peak

            # Ensure we have at least one sample
            if end <= start:
                end = start + 1

            # Extract MUAP segment
            muap_segment = signal_centered[start:end]

            # Duration in ms
            duration_ms = (end - start) * 1000 / fs
            durations.append(duration_ms)

            # Area under the curve (absolute value)
            area = np.sum(np.abs(muap_segment)) / fs  # Normalize by sampling rate
            areas.append(area)

        features["MUAP_MeanDuration"] = np.mean(durations)
        features["MUAP_MeanArea"] = np.mean(areas)

        # Firing rate (MUAPs per second)
        recording_duration = len(signal) / fs
        features["MUAP_FiringRate"] = len(peaks) / recording_duration

        return features
    except Exception:
        features["MUAP_Count"] = np.nan
        features["MUAP_MeanAmplitude"] = np.nan
        features["MUAP_MeanDuration"] = np.nan
        features["MUAP_MeanArea"] = np.nan
        features["MUAP_FiringRate"] = np.nan
        return features


def compute_fractal_dimension(signal: np.ndarray) -> float:
    """
    Petrosian's fractal dimension.
    """
    if len(signal) < 3:
        return np.nan

    try:
        diff = np.diff(signal)
        N_delta = np.sum(diff[1:] * diff[:-1] < 0)
        n = len(signal)
        return float(np.log10(n) / (np.log10(n) + np.log10(n / (n + 0.4 * N_delta))))
    except Exception:
        return np.nan


def compute_zero_crossings(signal: np.ndarray) -> int:
    """
    Count the number of zero crossings in the signal.
    """
    if len(signal) < 2:
        return 0

    signs = np.sign(signal)
    zc = np.sum(np.diff(signs) != 0)
    return int(zc)


def find_onset_time(signal: np.ndarray, fs: float = 1000, threshold: float = 0.01) -> float:
    """
    Find the onset time of the signal.
    """
    abs_signal = np.abs(signal)
    idx = np.where(abs_signal > threshold)[0]
    if len(idx) == 0:
        return np.nan
    return float(idx[0] / fs)


def compute_emg_statistical_features(
        emg_signal: np.ndarray,
        fs: float = 1000,
        muscle_name: Optional[str] = None
) -> Dict[str, float]:
    """
    Compute comprehensive statistical features from EMG signal for ONE repetition/window.
    Each feature is prefixed with <muscle_name> for clarity.

    Parameters:
    -----------
    emg_signal : np.ndarray
        Array of EMG values for a single window/repetition
    fs : float
        Sampling frequency in Hz
    muscle_name : str, optional
        Name of the muscle (e.g., 'del', 'trap')

    Returns:
    --------
    Dict[str, float]
        Dictionary of statistical features
    """
    features = {}
    prefix = f"{muscle_name}_" if muscle_name else ""
    N = len(emg_signal)

    # Edge case: if signal has no samples
    if N == 0:
        placeholders = [
            "Mean", "Median", "Mode", "StdDev", "Variance", "CV", "IQR", "Skewness", "Kurtosis",
            "Percentile_25", "Percentile_50", "Percentile_75", "Range_Q3_Q1", "Range_Q90_Q10",
            "Energy", "MeanPower", "TAV", "RMS", "Peak", "Min", "Range", "ARV", "ZeroCrossings",
            "MeanFrequency", "SNR", "Entropy", "FractalDimension", "CumulativeSum", "TimeToPeak",
            "OnsetTime",
            # Fatigue-specific features
            "MedianFrequency", "FrequencyRatio", "MuscleFatigueIndex", "SampleEntropy",
            "ConductionVelocity", "MUAP_Count", "MUAP_MeanAmplitude", "MUAP_MeanDuration",
            "MUAP_MeanArea", "MUAP_FiringRate"
        ]
        for p in placeholders:
            features[f"{prefix}{p}"] = np.nan
        return features

    # Convert to numpy array, handle missing values
    x = np.array(emg_signal, dtype=float)
    x = np.nan_to_num(x)  # if any NaNs, set to 0

    # Check sampling rate
    if fs <= 0:
        fs = 1000

    # Apply bandpass filter for EMG (20-450 Hz is typical for surface EMG)
    x_filtered = bandpass_filter_emg(x, fs=fs, lowcut=20, highcut=min(450, fs / 2 - 1), order=4)

    # 1) Central Tendency: Mean, Median, Mode
    mean_val = np.mean(x)
    features[f"{prefix}Mean"] = float(mean_val)

    median_val = np.median(x)
    features[f"{prefix}Median"] = float(median_val)

    hist, bin_edges = np.histogram(x, bins=50)
    if len(hist) > 0:
        mode_bin = np.argmax(hist)
        mode_est = 0.5 * (bin_edges[mode_bin] + bin_edges[mode_bin + 1])
    else:
        mode_est = mean_val
    features[f"{prefix}Mode"] = float(mode_est)

    # 2) Variability: StdDev, Variance, CV, IQR
    std_dev = np.std(x, ddof=1) if N > 1 else 0.0
    features[f"{prefix}StdDev"] = float(std_dev)

    var_val = std_dev ** 2
    features[f"{prefix}Variance"] = float(var_val)

    cv_val = (std_dev / mean_val) if np.abs(mean_val) > 1e-12 else np.nan
    features[f"{prefix}CV"] = float(cv_val)

    p25 = np.percentile(x, 25)
    p75 = np.percentile(x, 75)
    iqr_val = p75 - p25
    features[f"{prefix}IQR"] = float(iqr_val)

    # 3) Shape: Skewness, Kurtosis
    sk_val = skew(x, bias=False) if N > 1 else 0.0
    kt_val = kurtosis(x, bias=False) if N > 1 else 0.0
    features[f"{prefix}Skewness"] = float(sk_val)
    features[f"{prefix}Kurtosis"] = float(kt_val)

    # 4) Percentiles
    features[f"{prefix}Percentile_25"] = float(p25)
    features[f"{prefix}Percentile_50"] = float(median_val)  # same as median
    features[f"{prefix}Percentile_75"] = float(p75)
    features[f"{prefix}Range_Q3_Q1"] = float(iqr_val)  # same as IQR
    p10 = np.percentile(x, 10)
    p90 = np.percentile(x, 90)
    features[f"{prefix}Range_Q90_Q10"] = float(p90 - p10)

    # 5) Energy and Power
    energy_val = compute_signal_energy(x)
    features[f"{prefix}Energy"] = energy_val

    mean_power = compute_mean_power(x)
    features[f"{prefix}MeanPower"] = mean_power

    # TAV = sum of abs(x)
    tav_val = np.sum(np.abs(x))
    features[f"{prefix}TAV"] = float(tav_val)

    # RMS
    rms_val = np.sqrt(np.mean(x ** 2))
    features[f"{prefix}RMS"] = float(rms_val)

    # 6) Amplitude Features
    peak_val = np.max(x)
    min_val = np.min(x)
    features[f"{prefix}Peak"] = float(peak_val)
    features[f"{prefix}Min"] = float(min_val)
    features[f"{prefix}Range"] = float(peak_val - min_val)

    # ARV = Average Rectified Value = mean of abs(x)
    arv_val = np.mean(np.abs(x))
    features[f"{prefix}ARV"] = float(arv_val)

    # 7) Signal Stability
    # Zero Crossings
    zc_val = compute_zero_crossings(x)
    features[f"{prefix}ZeroCrossings"] = zc_val

    # Mean Frequency
    mf_val = compute_mean_frequency(x_filtered, fs)
    features[f"{prefix}MeanFrequency"] = mf_val

    # SNR
    snr_val = compute_snr(x)
    features[f"{prefix}SNR"] = snr_val

    # 8) Signal Complexity
    # Entropy (Shannon)
    hist_counts, bin_edges = np.histogram(x, bins=50, density=True)
    hist_counts = hist_counts + 1e-12  # avoid zero
    ent_val = entropy(hist_counts, base=2)  # Shannon entropy in bits
    features[f"{prefix}Entropy"] = float(ent_val)

    # Fractal Dimension
    fd_val = compute_fractal_dimension(x)
    features[f"{prefix}FractalDimension"] = fd_val

    # 9) Temporal Features
    # Cumulative Sum
    cumsum_val = np.sum(x)  # final value of cumsum
    features[f"{prefix}CumulativeSum"] = float(cumsum_val)

    # Time-to-Peak:
    idx_peak = np.argmax(x)
    time_to_peak = idx_peak / fs
    features[f"{prefix}TimeToPeak"] = float(time_to_peak)

    # Onset Detection:
    onset_time = find_onset_time(x, fs=fs, threshold=0.01 * peak_val)
    features[f"{prefix}OnsetTime"] = onset_time

    # 10) Fatigue-specific features (using filtered signal for frequency analysis)

    # Median Frequency (key fatigue indicator)
    median_freq = compute_median_frequency(x_filtered, fs)
    features[f"{prefix}MedianFrequency"] = median_freq

    # Frequency Ratio (low/high frequency power ratio)
    freq_ratio = compute_frequency_ratio(x_filtered, fs, low_cutoff=min(20, fs / 4),
                                         high_cutoff=min(250, fs / 2 - 1))
    features[f"{prefix}FrequencyRatio"] = freq_ratio

    # Muscle Fatigue Index
    mfi = compute_muscle_fatigue_index(x_filtered, fs)
    features[f"{prefix}MuscleFatigueIndex"] = mfi

    # 11) Advanced EMG metrics

    # Sample Entropy
    sample_ent = compute_sample_entropy(x, m=2, r=0.2)
    features[f"{prefix}SampleEntropy"] = sample_ent

    # Conduction Velocity Estimation
    cond_vel = estimate_conduction_velocity(x_filtered, fs)
    features[f"{prefix}ConductionVelocity"] = cond_vel

    # Motor Unit Action Potential (MUAP) Features
    muap_features = extract_muap_features(x_filtered, fs)
    for key, value in muap_features.items():
        features[f"{prefix}{key}"] = value

    return features


def compute_emg_statistical_features_multi_channel(
        emg_data: np.ndarray,
        sampling_rate: float,
        muscle_names: Optional[list] = None
) -> Dict[str, float]:
    """
    Compute statistical features for multi-channel EMG data.

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
        Dictionary of statistical features for all channels
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

        # Compute statistical features for this channel
        channel_features = compute_emg_statistical_features(
            channel_data, sampling_rate, muscle_name
        )
        features.update(channel_features)

    # Compute features for the mean across all channels (if multiple channels)
    if n_channels > 1:
        mean_emg = np.mean(emg_data, axis=0)
        mean_features = compute_emg_statistical_features(
            mean_emg, sampling_rate, "mean"
        )
        features.update(mean_features)

    return features


def compute_emg_statistical_features_for_window(
        window: Dict,
        fs_emg: float,
        muscle_names: Optional[list] = None
) -> Dict[str, float]:
    """
    Convenience function to extract statistical EMG features from a window dict.

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
        Dictionary of statistical EMG features
    """
    if "emg" not in window or window["emg"] is None:
        return {}

    emg_data = np.asarray(window["emg"])

    # Handle single channel case
    if emg_data.ndim == 1:
        emg_data = emg_data[np.newaxis, :]  # Add channel dimension

    return compute_emg_statistical_features_multi_channel(
        emg_data, fs_emg, muscle_names
    )


def compute_emg_statistical_features_per_repetition(
        emg_data: np.ndarray,
        repetition_segments: list,
        sampling_rate: float,
        muscle_names: Optional[list] = None
) -> Dict[int, Dict[str, float]]:
    """
    Compute statistical EMG features for each repetition segment.

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
        Dictionary mapping repetition number to statistical features
    """
    repetition_features = {}

    for rep_idx, (start_idx, end_idx) in enumerate(repetition_segments):
        # Extract data for this repetition
        rep_data = emg_data[:, start_idx:end_idx]

        if rep_data.shape[1] > 0:  # Ensure we have data
            # Compute statistical features for this repetition
            rep_features = compute_emg_statistical_features_multi_channel(
                rep_data, sampling_rate, muscle_names
            )