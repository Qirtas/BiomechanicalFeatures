from __future__ import annotations
import pandas as pd
from .config import Config, DEFAULT
from .preprocessing import preprocess_emg, preprocess_imu
from .windowing import make_windows
from .features_basic import compute_features_basic
from .segmentation import RepParams, segment_repetitions_from_gyr
from .windowing import make_windows, make_windows_from_segments
import numpy as np

def _validate_inputs(data, fs_emg, fs_imu):
    if ("emg" not in data) and ("imu" not in data):
        raise ValueError("data must contain 'emg' and/or 'imu'.")

    if data.get("emg") is not None:
        emg = np.asarray(data["emg"])
        if emg.ndim != 2:
            raise ValueError("EMG must be 2D array shaped (channels, samples).")
        if fs_emg is None:
            raise ValueError("fs_emg is required when EMG is provided.")

    imu = data.get("imu")
    if imu:
        acc = imu.get("acc")
        if acc is not None:
            acc = np.asarray(acc)
            if acc.shape[0] != 3:
                raise ValueError("IMU acc must be shaped (3, samples).")
            if fs_imu is None:
                raise ValueError("fs_imu is required when IMU is provided.")

# def extract_features(
#     data,
#     fs_emg: float | None = None,
#     fs_imu: float | None = None,
#     config: Config | None = None,
#     segmentation: dict | None = None,
#     muscle_names: list | None = None,
#     **overrides,   # lets users override small things without rebuilding a Config
# ):
#     """
#     data: {"emg": np.ndarray|None, "imu": {"acc": np.ndarray, "gyr": np.ndarray}|None}
#     """
#     cfg = (config or DEFAULT).with_overrides(**overrides)
#
#     emg = preprocess_emg(
#         data.get("emg"),
#         fs_emg,
#         band=cfg.emg.band,
#         order=cfg.emg.order,
#         notch_hz=cfg.emg.notch_hz,
#     ) if data.get("emg") is not None else None
#
#     imu = preprocess_imu(
#         data.get("imu"),
#         fs_imu,
#         cutoff_hz=cfg.imu.cutoff_hz,
#         order=cfg.imu.order,
#     ) if data.get("imu") is not None else None
#
#     # Decide windowing/segmentation
#     if segmentation and segmentation.get("mode") == "reps":
#         # repetition-based segmentation from gyr
#         if not imu or imu.get("gyr") is None or not fs_imu:
#             raise ValueError("Repetition segmentation requires IMU gyroscope and fs_imu.")
#         rp = RepParams(
#             axis=segmentation.get("axis"),
#             mode=segmentation.get("rep_mode", "peak_valley_pairs"),
#             smooth_s=segmentation.get("smooth_s", 0.20),
#             smooth_poly=segmentation.get("smooth_poly", 3),
#             min_prominence=segmentation.get("min_prominence", 0.20),
#             min_distance_s=segmentation.get("min_distance_s", 0.80),
#             site=segmentation.get("site"),
#         )
#         spans = segment_repetitions_from_gyr(imu["gyr"], fs_imu, rp)
#         windows = make_windows_from_segments(emg, imu, fs_emg, fs_imu, spans)
#     else:
#         # fallback/default: fixed windows (static holds, or when reps aren't desired)
#         windows = make_windows(
#             emg=emg, imu=imu, fs_emg=fs_emg, fs_imu=fs_imu,
#             window_s=cfg.window.window_s, step_s=cfg.window.step_s,
#             allow_partial=cfg.window.allow_partial,
#         )
#
#     rows = [compute_features_basic(w, fs_emg=fs_emg, fs_imu=fs_imu) for w in windows]
#     return pd.DataFrame(rows)

def extract_features(
    data,
    fs_emg: float | None = None,
    fs_imu: float | None = None,
    config: Config | None = None,
    segmentation: dict | None = None,
    muscle_names: list | None = None,  # Add this parameter
    imu_site: str | None = None,       # Add this parameter
    **overrides,   # lets users override small things without rebuilding a Config
):
    """
    data: {"emg": np.ndarray|None, "imu": {"acc": np.ndarray, "gyr": np.ndarray}|None}
    """
    cfg = (config or DEFAULT).with_overrides(**overrides)

    emg = preprocess_emg(
        data.get("emg"),
        fs_emg,
        band=cfg.emg.band,
        order=cfg.emg.order,
        notch_hz=cfg.emg.notch_hz,
    ) if data.get("emg") is not None else None

    imu = preprocess_imu(
        data.get("imu"),
        fs_imu,
        cutoff_hz=cfg.imu.cutoff_hz,
        order=cfg.imu.order,
    ) if data.get("imu") is not None else None

    # Decide windowing/segmentation
    if segmentation and segmentation.get("mode") == "reps":
        # repetition-based segmentation from gyr
        if not imu or imu.get("gyr") is None or not fs_imu:
            raise ValueError("Repetition segmentation requires IMU gyroscope and fs_imu.")
        rp = RepParams(
            axis=segmentation.get("axis"),
            mode=segmentation.get("rep_mode", "peak_valley_pairs"),
            smooth_s=segmentation.get("smooth_s", 0.20),
            smooth_poly=segmentation.get("smooth_poly", 3),
            min_prominence=segmentation.get("min_prominence", 0.20),
            min_distance_s=segmentation.get("min_distance_s", 0.80),
            site=segmentation.get("site"),
        )
        spans = segment_repetitions_from_gyr(imu["gyr"], fs_imu, rp)
        windows = make_windows_from_segments(emg, imu, fs_emg, fs_imu, spans)
    else:
        # fallback/default: fixed windows (static holds, or when reps aren't desired)
        windows = make_windows(
            emg=emg, imu=imu, fs_emg=fs_emg, fs_imu=fs_imu,
            window_s=cfg.window.window_s, step_s=cfg.window.step_s,
            allow_partial=cfg.window.allow_partial,
        )

    # Pass muscle_names and imu_site to compute_features_basic
    rows = [compute_features_basic(
        w,
        fs_emg=fs_emg,
        fs_imu=fs_imu,
        muscle_names=muscle_names,
        imu_site=imu_site
    ) for w in windows]
    return pd.DataFrame(rows)