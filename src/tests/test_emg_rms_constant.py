import numpy as np
from biomechfe import extract_features

def test_emg_rms_constant_signal():
    emg = np.ones((1, 2000)) * 2.0  # 2 s @ 1 kHz
    df = extract_features({"emg": emg}, fs_emg=1000)
    assert df["emg_rms_mean"].iloc[0] > 1.0  # loose check; preprocessing may reduce slightly
