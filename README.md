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
git clone <https://github.com/Qirtas/BiomechanicalFeatures.git>
cd biomech-features
pip install -e .
```

### Install from Test PyPI
The package is currently available on Test PyPI for team review and testing:

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ biomech-features```
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

### Expected File Structure

The library expects a hierarchical folder structure for automatic data loading:

```
<root>/
├── <movement>/
│   ├── emg_<muscle_name>/
│   │   └── *Subject_<id>*.csv
│   ├── <imu_site>/
│   │   ├── acc/
│   │   │   └── Subject_<id>_<site>_acc.csv
│   │   └── gyr/
│   │       └── Subject_<id>_<site>_gyr.csv
```

**Example:**
```
Dataset/
├── processed_data_35_i/
│   ├── emg_deltoideus_anterior/
│   │   └── Subject_1_deltoideus_anterior.csv
│   ├── emg_deltoideus_posterior/
│   │   └── Subject_1_deltoideus_posterior.csv
│   ├── Shoulder/
│   │   ├── acc/
│   │   │   └── Subject_1_Shoulder_acc.csv
│   │   └── gyr/
│   │       └── Subject_1_Shoulder_gyr.csv
```

### CSV File Format Requirements

#### EMG Files
EMG CSV files should contain a single column with the signal data:
- **Preferred column names**: `"EMG"`, `"emg"`, `"Signal"`, `"signal"`, `"value"`, `"amplitude"`
- **Fallback**: First numeric column will be used
- **Data type**: Numeric values (float)

```csv
EMG
0.1234
0.1456
0.1123
...
```

#### IMU Files
IMU CSV files should contain 3-axis data (X, Y, Z):
- **Preferred column names**: 
  - Accelerometer: `"Acc_X"`, `"Acc_Y"`, `"Acc_Z"`
  - Gyroscope: `"Gyr_X"`, `"Gyr_Y"`, `"Gyr_Z"`
- **Fallback options**: `"X"`, `"Y"`, `"Z"` or `"x"`, `"y"`, `"z"`
- **Data type**: Numeric values (float)

```csv
Acc_X,Acc_Y,Acc_Z
9.81,0.12,-0.34
9.85,0.15,-0.31
9.79,0.09,-0.37
...
```

### Input Data Structure (In-Memory)

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

### Adapting Your Data Format

If your data doesn't match the expected format, here are common adaptation strategies:

#### 1. Different File Structure

```python
import pandas as pd
import numpy as np
from biomechfe import extract_features

# Load your custom format
def load_custom_format(file_path):
    df = pd.read_csv(file_path)
    
    # Example: Convert single file with multiple EMG channels
    emg_columns = ["muscle1", "muscle2", "muscle3"]
    emg_data = df[emg_columns].values.T  # Transpose to (channels, samples)
    
    # Example: Convert IMU data with different column names
    imu_data = {
        "acc": df[["accel_x", "accel_y", "accel_z"]].values.T,
        "gyr": df[["gyro_x", "gyro_y", "gyro_z"]].values.T
    }
    
    return {"emg": emg_data, "imu": imu_data}

# Use with feature extraction
data = load_custom_format("your_file.csv")
df = extract_features(data, fs_emg=1000, fs_imu=100)
```

#### 2. Different Column Names

```python
def adapt_column_names(df, format_type="emg"):
    """Adapt your column names to library expectations"""
    
    if format_type == "emg":
        # Map your column names to expected ones
        column_mapping = {
            "signal_amplitude": "EMG",
            "muscle_activity": "EMG"
        }
        df = df.rename(columns=column_mapping)
    
    elif format_type == "imu_acc":
        column_mapping = {
            "acceleration_x": "Acc_X",
            "acceleration_y": "Acc_Y", 
            "acceleration_z": "Acc_Z"
        }
        df = df.rename(columns=column_mapping)
    
    elif format_type == "imu_gyr":
        column_mapping = {
            "angular_vel_x": "Gyr_X",
            "angular_vel_y": "Gyr_Y",
            "angular_vel_z": "Gyr_Z"
        }
        df = df.rename(columns=column_mapping)
    
    return df
```

#### 3. Single File with Multiple Modalities

```python
def split_multimodal_file(file_path):
    """Split a single file containing both EMG and IMU data"""
    df = pd.read_csv(file_path)
    
    # Extract EMG channels
    emg_cols = [col for col in df.columns if 'emg' in col.lower()]
    emg_data = df[emg_cols].values.T
    
    # Extract IMU channels
    acc_cols = [col for col in df.columns if 'acc' in col.lower()]
    gyr_cols = [col for col in df.columns if 'gyr' in col.lower()]
    
    imu_data = {
        "acc": df[acc_cols].values.T,
        "gyr": df[gyr_cols].values.T
    }
    
    return {
        "emg": emg_data,
        "imu": imu_data
    }
```

#### 4. Different Sampling Rates

```python
from scipy import signal

def resample_data(data, original_fs, target_fs):
    """Resample data to match expected sampling rate"""
    if original_fs == target_fs:
        return data
    
    # Calculate resampling ratio
    num_samples = int(data.shape[-1] * target_fs / original_fs)
    
    if data.ndim == 1:
        return signal.resample(data, num_samples)
    else:
        # For multi-channel data
        resampled = np.zeros((data.shape[0], num_samples))
        for i in range(data.shape[0]):
            resampled[i] = signal.resample(data[i], num_samples)
        return resampled

# Example usage
emg_resampled = resample_data(emg_data, original_fs=2000, target_fs=1000)
```

#### 5. Custom Data Loader Function

```python
def create_custom_loader(base_path, subject_id, trial_name):
    """
    Create a custom loader for your specific data format
    Returns data in library-compatible format
    """
    
    # Load EMG data (example: single file with multiple muscles)
    emg_file = f"{base_path}/{trial_name}/emg_data_subject_{subject_id}.csv"
    emg_df = pd.read_csv(emg_file)
    
    # Assuming columns: timestamp, muscle1, muscle2, muscle3
    emg_data = emg_df[["muscle1", "muscle2", "muscle3"]].values.T
    
    # Load IMU data (example: separate files for acc and gyr)
    acc_file = f"{base_path}/{trial_name}/accelerometer_subject_{subject_id}.csv"
    gyr_file = f"{base_path}/{trial_name}/gyroscope_subject_{subject_id}.csv"
    
    acc_df = pd.read_csv(acc_file)
    gyr_df = pd.read_csv(gyr_file)
    
    imu_data = {
        "acc": acc_df[["x", "y", "z"]].values.T,
        "gyr": gyr_df[["x", "y", "z"]].values.T
    }
    
    return {
        "emg": emg_data,
        "imu": imu_data,
        "fs": {"emg": 1000, "imu": 100},
        "meta": {
            "subject": subject_id,
            "trial": trial_name,
            "emg_channels": ["muscle1", "muscle2", "muscle3"]
        }
    }

# Usage
data = create_custom_loader("path/to/data", subject_id=1, trial_name="trial1")
df = extract_features(data, fs_emg=data["fs"]["emg"], fs_imu=data["fs"]["imu"])
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

### Development Setup

```bash
git clone <https://github.com/Qirtas/BiomechanicalFeatures.git>
cd biomech-features
pip install -e ".[dev]"
```


---

## Citation

If you use this library in your research, please cite:

---

## Contact

- **Author**: Malik Qirtas
- **Email**: mqirtas@ucc.ie
- **Issues**: [GitHub Issues](https://https://github.com/Qirtas/BiomechanicalFeatures/issues)