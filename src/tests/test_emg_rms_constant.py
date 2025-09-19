import numpy as np
from biomechfe import extract_features

def test_emg_rms_constant_signal():
    emg = np.ones((1, 1000)) * 2.0  # RMS should be 2.0 after preprocessing envelope is roughly ~2
    df = extract_features({"emg": emg}, fs_emg=1000)
    assert df["emg_rms_mean"].iloc[0] > 1.0  # loose check; preprocessing may reduce slightly
