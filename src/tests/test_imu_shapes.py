import numpy as np
import pytest
from biomechfe import extract_features

def test_bad_imu_shape_raises():
    acc = np.random.randn(2, 100)  # wrong first dim
    with pytest.raises(ValueError):
        extract_features({"imu": {"acc": acc}}, fs_imu=100)
