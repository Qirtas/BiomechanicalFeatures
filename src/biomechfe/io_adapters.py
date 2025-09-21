# src/biomechfe/io_adapters.py
from __future__ import annotations
import os, glob
import numpy as np
from typing import Iterable, Literal, Sequence
import pandas as pd

def _load_csv_matrix(path: str, cols: list[str]) -> np.ndarray:
    df = pd.read_csv(path)
    # Accept flexible column casing; fall back if needed
    present = [c for c in cols if c in df.columns]
    if len(present) != len(cols):
        raise ValueError(f"Expected columns {cols} in {path}, got {list(df.columns)[:8]}...")
    arr = df[cols].to_numpy(dtype=float).T   # (channels, samples)
    return arr


def _load_csv_matrix(path: str, expected: list[str]) -> np.ndarray:
    """
    Load CSV into (channels, samples).
    expected: preferred column names (e.g. ["Acc_X", "Acc_Y", "Acc_Z"]).
    Falls back to common variants like ["X","Y","Z"].
    """
    df = pd.read_csv(path)

    # Try exact expected columns first
    if all(col in df.columns for col in expected):
        cols = expected
    # Try bare X/Y/Z
    elif all(col in df.columns for col in ["X", "Y", "Z"]):
        cols = ["X", "Y", "Z"]
    # Try lowercase x/y/z
    elif all(col in df.columns for col in ["x", "y", "z"]):
        cols = ["x", "y", "z"]
    else:
        raise ValueError(
            f"Could not find expected IMU columns in {path}. "
            f"Got {list(df.columns)[:10]}"
        )

    arr = df[cols].to_numpy(dtype=float).T  # shape (3, n_samples)
    return arr

# ---------- helpers ----------

def _first_numeric_series(df: pd.DataFrame) -> np.ndarray:
    num_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
    if not num_cols:
        raise ValueError("No numeric column found.")
    return df[num_cols[0]].to_numpy(dtype=float)

def _load_emg_signal(path: str) -> np.ndarray:
    df = pd.read_csv(path)
    # try common names, fall back to first numeric
    for col in ("EMG", "emg", "Signal", "signal", "value", "amplitude"):
        if col in df.columns:
            return df[col].to_numpy(dtype=float)
    return _first_numeric_series(df)

def _pad_to_len_1d(x: np.ndarray, target_len: int) -> np.ndarray:
    if x.shape[0] >= target_len:
        return x[:target_len]
    pad_val = float(x[-1]) if x.size else 0.0
    pad = np.full(target_len - x.shape[0], pad_val, dtype=float)
    return np.concatenate([x, pad], axis=0)

def _load_xyz_matrix(path: str, prefer: Sequence[str]) -> np.ndarray:
    df = pd.read_csv(path)
    # try preferred names first (e.g., ["Acc_X","Acc_Y","Acc_Z"] or ["Gyr_X","Gyr_Y","Gyr_Z"])
    if all(c in df.columns for c in prefer):
        cols = list(prefer)
    elif all(c in df.columns for c in ["X", "Y", "Z"]):
        cols = ["X", "Y", "Z"]
    elif all(c in df.columns for c in ["x", "y", "z"]):
        cols = ["x", "y", "z"]
    else:
        raise ValueError(f"Could not find XYZ columns in {path}. Got: {list(df.columns)[:10]}")
    return df[cols].to_numpy(dtype=float).T  # (3, n)

def _autodetect_subdirs(parent: str, prefix: str | None = None) -> list[str]:
    if not os.path.isdir(parent):
        return []
    dirs = [d for d in os.listdir(parent) if os.path.isdir(os.path.join(parent, d))]
    if prefix:
        dirs = [d for d in dirs if d.lower().startswith(prefix.lower())]
    return sorted(dirs)

# ---------- main loader ----------

def load_record_from_hierarchy(
    root: str,
    movement: str,                 # e.g., "35Internal/processed_data_35_i"
    subject_id: int,
    *,
    # modality toggles
    load_emg: bool = True,
    load_imu: bool = True,
    # EMG options
    emg_dirs: Iterable[str] | None = None,   # e.g., ["emg_deltoideus_anterior", ...]; if None -> autodetect "emg_*"
    emg_align: Literal["trim", "pad"] = "trim",
    fs_emg: float | None = 1000.0,
    # IMU options
    imu_sites: Iterable[str] | None = None,  # e.g., ["Shoulder", "Upperarm"]; if None -> autodetect common sites
    imu_kinds: Iterable[str] = ("acc", "gyr"),  # which kinds to load
    fs_imu: float | None = 100.0,
) -> dict:
    """
    Build a BiomechRecord for one subject/movement from a hierarchical folder structure.

    Expected on disk (flexible):
      <root>/<movement>/EMG-like folders (default autodetect 'emg_*') with files matching '*Subject_<id>*.csv'
      <root>/<movement>/<imu_site>/<kind>/Subject_<id>_<imu_site>_<kind>.csv

    Returns dict: {"emg", "imu", "fs", "meta"} following the library's standard in-memory schema.
    """
    mv_dir = os.path.join(root, movement)

    # ----- EMG -----
    emg_mat = None
    emg_names: list[str] = []
    if load_emg:
        # autodetect EMG folders if not provided (any subdir starting with 'emg_')
        if emg_dirs is None:
            emg_dirs = [d for d in _autodetect_subdirs(mv_dir) if d.lower().startswith("emg_")]
        emg_rows: list[np.ndarray] = []
        lengths: list[int] = []
        for m in emg_dirs:
            pattern = os.path.join(mv_dir, m, f"*Subject_{subject_id}*.csv")
            files = glob.glob(pattern)
            if not files:
                continue
            sig = _load_emg_signal(files[0])
            emg_rows.append(sig)
            lengths.append(sig.shape[0])
            emg_names.append(m)
        if emg_rows:
            min_len, max_len = int(np.min(lengths)), int(np.max(lengths))
            if min_len != max_len:
                print(f"[io_adapters] EMG lengths differ (min={min_len}, max={max_len}) → {emg_align}.")
            aligned = [r[:min_len] if emg_align == "trim" else _pad_to_len_1d(r, max_len) for r in emg_rows]
            target = min_len if emg_align == "trim" else max_len
            emg_mat = np.vstack([a[np.newaxis, :] for a in aligned]).astype(float)  # (n_ch, target)

    # ----- IMU -----
    imu_dict: dict = {"acc": None, "gyr": None, "mag": None}
    loaded_sites: list[str] = []
    if load_imu:
        # autodetect typical sites if not provided
        if imu_sites is None:
            candidate = ["Shoulder", "Upperarm", "Forearm", "Hand", "Torso"]
            imu_sites = [s for s in candidate if os.path.isdir(os.path.join(mv_dir, s))]
            if not imu_sites:
                # fallback: any subdir at mv_dir with acc/ or gyr/ inside
                imu_sites = []
                for d in _autodetect_subdirs(mv_dir):
                    if os.path.isdir(os.path.join(mv_dir, d, "acc")) or os.path.isdir(os.path.join(mv_dir, d, "gyr")):
                        imu_sites.append(d)
        # load the first available site that has data (simple starter; extend as needed)
        for site in imu_sites:
            site_dir = os.path.join(mv_dir, site)
            got_any = False
            acc_path = os.path.join(site_dir, "acc", f"Subject_{subject_id}_{site}_acc.csv")
            gyr_path = os.path.join(site_dir, "gyr", f"Subject_{subject_id}_{site}_gyr.csv")
            if "acc" in imu_kinds and os.path.exists(acc_path):
                imu_dict["acc"] = _load_xyz_matrix(acc_path, ["Acc_X", "Acc_Y", "Acc_Z"])
                got_any = True
            if "gyr" in imu_kinds and os.path.exists(gyr_path):
                imu_dict["gyr"] = _load_xyz_matrix(gyr_path, ["Gyr_X", "Gyr_Y", "Gyr_Z"])
                got_any = True
            if got_any:
                loaded_sites.append(site)
                break  # pick one site for a single record; change if you want multi-site support

    return {
        "emg": emg_mat,
        "imu": imu_dict if load_imu else {"acc": None, "gyr": None, "mag": None},
        "fs": {"emg": fs_emg if load_emg else None, "imu": fs_imu if load_imu else None},
        "meta": {
            "subject": subject_id,
            "movement": movement,
            "imu_site": loaded_sites[0] if loaded_sites else None,
            "emg_channels": emg_names,
            "path": mv_dir,
        },
    }
