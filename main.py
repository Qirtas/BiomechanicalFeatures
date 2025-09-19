import numpy as np
from biomechfe import extract_features

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    emg = np.random.randn(4, 2000)  # 4 channels, 2s @ 1000 Hz
    acc = np.random.randn(3, 200)  # 3 axes, 2s @ 100 Hz

    df = extract_features({"emg": emg, "imu": {"acc": acc}}, fs_emg=1000, fs_imu=100)
    print(df.head())

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
