import edesdetectrl.dataloaders.echonet.label_frames as lf
import numpy as np


def test_previous_and_next_maximum():
    # Simple case
    # Index:           1        4     6
    arr = np.array([2, 3, 2, 1, 0, 1, 2, 1, 0])
    assert lf.previous_peak(arr, 4) == 1
    assert lf.next_peak(arr, 4) == 6

    # Plateaus return the nearest element in the plateau
    # Index:           1        4     6
    arr = np.array([3, 3, 2, 1, 0, 1, 2, 2, 2])
    assert lf.previous_peak(arr, 4) == 1
    assert lf.next_peak(arr, 4) == 6

    arr = np.array([3, 2, 1, 0])
    assert lf.previous_peak(arr, 3) == 0
    arr = np.array([0, 1, 2, 3])
    assert lf.next_peak(arr, 0) == 3

    arr = np.array([0])
    assert lf.previous_peak(arr, 0) == 0
    assert lf.next_peak(arr, 0) == 0
