# biomech-features

A Python library for extracting biomechanical features from IMU (Inertial Measurement Unit) and EMG (Electromyography) sensor data. This library provides simple, fast feature extraction with filtering, segmentation (fixed windows or repetition-based), and feature computation into a tidy pandas DataFrame.

---

## Features

- **Multi-modal sensor support**: EMG and IMU (accelerometer + gyroscope) data processing
- **Flexible segmentation**: Fixed time windows or repetition-based segmentation
- **Comprehensive feature extraction**: 100+ features across multiple domains (time, frequency, statistical)
- **Easy integration**: Simple API with pandas DataFrame output
- **Extensible**: Modular design for adding custom features

---

## Installation

### Requirements
- Python ≥ 3.9
- NumPy
- SciPy  
- Pandas

### Install from source
```bash
git clone <repository-url>
cd biomech-features
pip install -e .
```

### Install from PyPI (coming soon)
```bash
pip install biomech-features
```

---

## Quick Start

### Basic Usage

```python
from biomechfe import extract_features, load_record_from_hierarchy

# Load your data
rec = load_record_from_hierarchy(
    root_path, subject_id, trial_id, 
    load_emg=True, load_imu=True
)

# Extract features from IMU data
df_imu = extract_features(
    {"imu": rec["imu"]},
    fs_imu=rec["fs"]["imu"],
    window_s=2.0,
    step_s=1.0
)

# Extract features from EMG data
df_emg = extract_features(
    {"emg": rec["emg"]},
    fs_emg=rec["fs"]["emg"],
    muscle_names=["deltoid", "biceps"],
    window_s=1.0,
    step_s=0.5
)
```

### Repetition-based Segmentation

```python
# Segment data based on movement repetitions
df = extract_features(
    {"imu": rec["imu"]},
    fs_imu=rec["fs"]["imu"],
    segmentation={
        "mode": "reps",
        "rep_mode": "zero_between_valley_peak",
        "axis": None,  # auto-select gyroscope axis
        "min_prominence": 0.25,
        "min_distance_s": 0.8,
        "smooth_s": 0.20,
    }
)
```

### Single Muscle Analysis

```python
# Extract features from specific muscle
target_muscle = "emg_deltoideus_posterior"
emg_channels = rec["meta"]["emg_channels"]
muscle_idx = emg_channels.index(target_muscle)

single_muscle_emg = rec["emg"][muscle_idx:muscle_idx + 1, :]
muscle_name = target_muscle.replace('emg_', '')

df = extract_features(
    {"emg": single_muscle_emg},
    fs_emg=rec["fs"]["emg"],
    muscle_names=[muscle_name],
    window_s=3.0,
    step_s=3.0
)
```

---

## Feature Categories

### EMG Features

| Category | Description | Sub-features |
|----------|-------------|--------------|
| **Integrated Features** | Time-domain integration measures | Integrated EMG (iEMG), Modified iEMG |
| **Mean Absolute Value (MAV)** | Amplitude-based features | MAV, MAV1, MAV2, Enhanced MAV |
| **Statistical Features** | Statistical descriptors | RMS, Variance, Standard Deviation, Skewness, Kurtosis |
| **Frequency Features (STFT)** | Short-time Fourier transform | Mean/Median Power Frequency, Spectral Moments, Power Ratios |
| **Wavelength Features** | Signal complexity measures | Waveform Length, Average Wavelength |
| **Zero Crossing (ZC)** | Signal crossing analysis | Zero Crossings, Slope Sign Changes, Willison Amplitude |

### IMU Accelerometer Features

| Category | Description | Sub-features |
|----------|-------------|--------------|
| **3-Axis Features** | Basic acceleration measures | Mean, RMS, Standard Deviation, Range per axis |
| **Magnitude Features** | Vector magnitude analysis | Mean/RMS/Std of magnitude, Peak acceleration |
| **Jerk Features** | Rate of acceleration change | Jerk magnitude, Jerk RMS, Peak jerk |
| **Movement Frequency** | Frequency domain analysis | Dominant frequencies, Power spectral density |
| **MPSD Features** | Motion-specific power spectral density | Frequency band power ratios |
| **RT Variability** | Real-time variability measures | Coefficient of variation, Variability indices |

### IMU Gyroscope Features

| Category | Description | Sub-features |
|----------|-------------|--------------|
| **3-Axis Angular Features** | Angular velocity measures | Mean, RMS, Standard Deviation, Range per axis |
| **Range of Motion (ROM)** | Movement extent analysis | Angular displacement, Total ROM |
| **Movement Frequency** | Rotational frequency analysis | Dominant frequencies, Spectral characteristics |
| **MPSD Features** | Angular motion power analysis | Frequency band distributions |
| **RT Variability** | Angular velocity variability | Smoothness indices, Consistency measures |

---

## API Reference

### Main Function

#### `extract_features(data, **kwargs)`

Extract features from sensor data.

**Parameters:**
- `data` (dict): Dictionary containing sensor data
  - `"emg"`: EMG data array (n_channels, n_samples)
  - `"imu"`: IMU data dict with `"acc"` and `"gyr"` keys
- `fs_emg` (float, optional): EMG sampling frequency
- `fs_imu` (float, optional): IMU sampling frequency
- `muscle_names` (list, optional): Names of EMG channels
- `imu_site` (str, optional): IMU sensor location name
- `window_s` (float, default=2.0): Window size in seconds
- `step_s` (float, default=1.0): Step size in seconds
- `extended_acc` (bool, default=False): Use extended accelerometer features
- `segmentation` (dict, optional): Repetition-based segmentation parameters

**Returns:**
- `pandas.DataFrame`: Features with metadata (window indices, timing, etc.)

### Segmentation Modes

#### Fixed Windows
```python
# Default: fixed-size sliding windows
extract_features(data, window_s=2.0, step_s=1.0)
```

#### Repetition-based
```python
segmentation = {
    "mode": "reps",
    "rep_mode": "zero_between_valley_peak",  # or "zero_between_peak_valley"
    "axis": None,  # auto-select or specify 0/1/2
    "min_prominence": 0.25,
    "min_distance_s": 0.8,
    "smooth_s": 0.20,
}
```

---

## Data Format

### Input Data Structure

```python
# EMG data: (n_channels, n_samples)
emg_data = np.array([[...], [...]])  # 2D array

# IMU data: dictionary with accelerometer and gyroscope
imu_data = {
    "acc": np.array([[...], [...], [...]]),  # (3, n_samples) - x,y,z
    "gyr": np.array([[...], [...], [...]]),  # (3, n_samples) - x,y,z
}

# Complete data structure
data = {
    "emg": emg_data,
    "imu": imu_data
}
```

### Output DataFrame

The output DataFrame includes:
- **Feature columns**: All computed features with descriptive names
- **Metadata columns**: 
  - `window_idx`: Window/segment index
  - `start_time`: Window start time (seconds)
  - `end_time`: Window end time (seconds)
  - `duration`: Window duration (seconds)

---

## Examples

### Complete Workflow Example

```python
import numpy as np
import pandas as pd
from biomechfe import extract_features

# Load your data (example with synthetic data)
fs_emg = 1000  # Hz
fs_imu = 100   # Hz
duration = 10  # seconds

# Create synthetic EMG data (2 channels)
emg = np.random.randn(2, fs_emg * duration)

# Create synthetic IMU data
imu = {
    "acc": np.random.randn(3, fs_imu * duration),
    "gyr": np.random.randn(3, fs_imu * duration)
}

# Extract features
df = extract_features(
    {"emg": emg, "imu": imu},
    fs_emg=fs_emg,
    fs_imu=fs_imu,
    muscle_names=["biceps", "triceps"],
    imu_site="wrist",
    window_s=2.0,
    step_s=1.0,
    extended_acc=True
)

# Save results
df.to_csv("extracted_features.csv", index=False)
print(f"Extracted {len(df.columns)} features from {len(df)} windows")
```

### Processing Multiple Subjects

```python
results = []

for subject_id in subject_list:
    for trial_id in trial_list:
        # Load data
        rec = load_record_from_hierarchy(
            root_path, subject_id, trial_id,
            load_emg=True, load_imu=True
        )
        
        # Extract features
        df = extract_features(
            {"emg": rec["emg"], "imu": rec["imu"]},
            fs_emg=rec["fs"]["emg"],
            fs_imu=rec["fs"]["imu"],
            muscle_names=rec["meta"]["emg_channels"],
            window_s=2.0
        )
        
        # Add metadata
        df["subject_id"] = subject_id
        df["trial_id"] = trial_id
        results.append(df)

# Combine all results
final_df = pd.concat(results, ignore_index=True)
```

---

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

### Development Setup

```bash
git clone <repository-url>
cd biomech-features
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest tests/
```

---

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

## Citation

If you use this library in your research, please cite:

```bibtex
@software{biomech_features,
  title={biomech-features: Feature extraction for IMU and EMG signals},
  author={Malik Qirtas},
  year={2024},
  url={https://github.com/your-username/biomech-features}
}
```

---

## Contact

- **Author**: Malik Qirtas
- **Email**: mqirtas@ucc.ie
- **Issues**: [GitHub Issues](https://https://github.com/Qirtas/BiomechanicalFeatures/issues)