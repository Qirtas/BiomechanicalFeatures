import numpy as np
from biomechfe import extract_features

def test_smoke_runs():
    emg = np.random.randn(2, 1000)
    acc = np.random.randn(3, 100)
    df = extract_features({"emg": emg, "imu": {"acc": acc}}, fs_emg=1000, fs_imu=100)
    assert not df.empty
