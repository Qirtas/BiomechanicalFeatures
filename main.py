import numpy as np
from biomechfe import extract_features
from biomechfe.config import Config, EMGSettings, IMUSettings, WindowSettings
from biomechfe.io_adapters import load_record_from_hierarchy
from biomechfe.config import DEFAULT

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    root = "src/Dataset/"

    # IMU Only
    rec = load_record_from_hierarchy(root, "processed_data_35_i", 1,
                                     load_emg=True, load_imu=True)

    print("EMG data shape:", rec["emg"].shape if rec["emg"] is not None else "None")
    print("EMG data type:", type(rec["emg"]))
    print("EMG channels:", rec["meta"]["emg_channels"])

    if rec["emg"] is not None:
        print("EMG data sample:", rec["emg"][:, :5])

    df = extract_features(
        {"imu": rec["imu"]},
        fs_imu=rec["fs"]["imu"],
        extended_acc=True,  # This would be a new parameter you add
        segmentation={
            "mode": "reps",
            "rep_mode": "zero_between_valley_peak",  # internal rotation
            # or "zero_between_peak_valley"           # external rotation
            "axis": None,  # auto-pick gyro axis by variance; or set 0/1/2
            "min_prominence": 0.25,
            "min_distance_s": 0.8,  # ~50 bpm cadence ≈ 1.2 s; tune per task
            "smooth_s": 0.20,
        }
    )


    # EMG Only

    # Extract only deltoideus posterior
    target_muscle = "emg_deltoideus_posterior"
    emg_channels = rec["meta"]["emg_channels"]
    muscle_idx = emg_channels.index(target_muscle)

    # Extract single channel (keep 2D shape)
    single_muscle_emg = rec["emg"][muscle_idx:muscle_idx + 1, :]
    muscle_name = target_muscle.replace('emg_', '')

    print(f"Single muscle EMG shape: {single_muscle_emg.shape}")
    print(f"Muscle name: {muscle_name}")

    df = extract_features(
        {"emg": single_muscle_emg},
        fs_emg=rec["fs"]["emg"],
        muscle_names=[muscle_name],
        window_s=3.0,
        step_s=3.0
    )

    print("Features extracted successfully!")
    print("DataFrame shape:", df.shape)

    df.to_csv("features_extracted_EMG.csv", index=False)
    print(rec["meta"])
    print(df.head())

