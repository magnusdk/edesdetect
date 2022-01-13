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


def _test_label_frames_helper(ed, es):
    def get_image(t):
        """A slightly noisy frame of a video that repeats every 2*pi t."""
        size = 100
        image = np.linspace(0, 4 * 2 * np.pi, size)
        image = np.tile(image, (size, 1))
        # image *= (image.T / 3) ** 2
        image = np.sin(image + t)
        image += np.random.rand(size, size)
        return image

    # Video repeats 5 times, once every 20 frames.
    # That means that if ED is at frame i, then frame i+10 will be ES, i.e.: half a cycle afterwards.
    video = np.array([get_image(c) for c in np.linspace(0, 5 * 2 * np.pi, 100)])
    a = 0 if ed < es else 1

    def assert_ground_truth_start_alignment(gt, start):
        # Just before and after ED
        assert gt[ed - 1 - start] == 0, "The frame just before ED is diastolic"
        assert gt[ed - start] == 0, "ED is diastolic"
        assert gt[ed + 1 - start] == 1, "The frame just after ED is systolic"
        # Just before and after ES
        assert gt[es - 1 - start] == 1, "The frame just before ES is systolic"
        assert gt[es - start] == 1, "ES is systolic"
        assert gt[es + 1 - start] == 0, "The frame just after ES is diastolic"

    # Using p=1 is very optimistic and labels all frames to and including the next (or previous) event.
    gt, start, end = lf.label_frames(video, ed, es, p=1)
    # Even though 30 is ES, it is included as diastolic when p=1.
    assert start == 30
    assert end == 60
    assert gt == [a] * 11 + [1 - a] * 10 + [a] * 10
    assert_ground_truth_start_alignment(gt, start)

    # Using p=0.5 labels halfways between difference peaks.
    gt, start, end = lf.label_frames(video, ed, es, p=0.5)
    assert start == 35
    assert end == 55
    assert gt == [a] * 6 + [1 - a] * 10 + [a] * 5
    assert_ground_truth_start_alignment(gt, start)

    # The half period is 10, so using p=0.1 would label only 1 additional frame on either side.
    gt, start, end = lf.label_frames(video, ed, es, p=0.1)
    assert start == 39
    assert end == 51
    assert gt == [a] * 2 + [1 - a] * 10 + [a]
    assert_ground_truth_start_alignment(gt, start)


def test_label_frames():
    _test_label_frames_helper(40, 50)
    _test_label_frames_helper(50, 40)