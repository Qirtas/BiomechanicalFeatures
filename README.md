Input: {"emg": (ch, n_emg), "imu": {"acc": (3, n_imu)}}, plus fs_emg, fs_imu.

Output: Pandas DataFrame, one row per window, columns = feature names.

Defaults: window_s=2.0, step_s=0.5.