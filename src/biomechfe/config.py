# src/biomechfe/config.py
from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Optional, Tuple, List, Dict, Any

# --- leaf settings ---

@dataclass(frozen=True)
class EMGSettings:
    band: Tuple[float, float] = (30.0, 350.0)   # Hz
    order: int = 4
    notch_hz: Optional[List[float]] = None      # e.g., [50.0] or [60.0]

@dataclass(frozen=True)
class IMUSettings:
    cutoff_hz: float = 20.0     # Hz (low-pass for acc/gyr)
    order: int = 4
    use_mag: bool = False       # reserved for later

@dataclass(frozen=True)
class WindowSettings:
    window_s: float = 2.0
    step_s: float = 0.5
    allow_partial: bool = False # if True, emit a single short window when clip < window

# --- top-level config ---

@dataclass(frozen=True)
class Config:
    emg: EMGSettings = EMGSettings()
    imu: IMUSettings = IMUSettings()
    window: WindowSettings = WindowSettings()

    # ergonomic override utilities
    def with_overrides(self, **kw: Any) -> "Config":
        """
        Supported keys:
          emg_band, emg_order, emg_notch
          imu_cutoff, imu_order, imu_use_mag
          window_s, step_s, allow_partial
        """
        c = self
        if "emg_band" in kw or "emg_order" in kw or "emg_notch" in kw:
            c = replace(
                c,
                emg=replace(
                    c.emg,
                    band=kw.get("emg_band", c.emg.band),
                    order=kw.get("emg_order", c.emg.order),
                    notch_hz=kw.get("emg_notch", c.emg.notch_hz),
                ),
            )
        if "imu_cutoff" in kw or "imu_order" in kw or "imu_use_mag" in kw:
            c = replace(
                c,
                imu=replace(
                    c.imu,
                    cutoff_hz=kw.get("imu_cutoff", c.imu.cutoff_hz),
                    order=kw.get("imu_order", c.imu.order),
                    use_mag=kw.get("imu_use_mag", c.imu.use_mag),
                ),
            )
        if "window_s" in kw or "step_s" in kw or "allow_partial" in kw:
            c = replace(
                c,
                window=replace(
                    c.window,
                    window_s=kw.get("window_s", c.window.window_s),
                    step_s=kw.get("step_s", c.window.step_s),
                    allow_partial=kw.get("allow_partial", c.window.allow_partial),
                ),
            )
        return c

# Presets we can expand
DEFAULT = Config()
DYNAMIC_TASKS = DEFAULT  # same processing by default
STATIC_TASKS  = DEFAULT  # kept for semantic clarity; tweak if you ever diverge
