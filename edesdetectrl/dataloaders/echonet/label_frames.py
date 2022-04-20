from typing import List, Tuple

import numpy as np
from edesdetectrl.dataloaders import GroundTruth
from scipy.ndimage import gaussian_filter1d


def next_peak(x, start_i):
    """Return the index of the next peak (local maximum) after start_i in x."""
    grad = x[1:] - x[:-1]
    max_diff_i = start_i
    for i in range(start_i + 1, len(x) - 1):
        max_diff_i = i
        if grad[i] <= 0:
            # The first frame that is not increasing is the closest next maximum (peak)
            return max_diff_i
    return len(x) - 1


def previous_peak(x, start_i):
    """Return the index of the previous peak (local maximum) before start_i in x."""
    grad = x[1:] - x[:-1]
    max_diff_i = start_i
    for i in range(start_i - 1, 0, -1):
        max_diff_i = i
        if grad[i - 1] >= 0:
            # The first frame that is not decreasing when searching backwards is the closest previous maximum (peak)
            return max_diff_i
    return 0


def label_frames(video, ed_i, es_i, p=0.75) -> Tuple[List[GroundTruth], int, int]:
    """
    Grab some frames before first keyframe (either ED or ES) where we are sure of the phase.

    We can be relatively sure of the phase by looking at the difference between one of the
    keyframes (either ED or ES) and all other frames. The frames leading up to the previous
    frame with the most difference from the first keyframe will have the same phase as the
    keyframe, and likewise the next frame with the most difference from the last keyframe
    will have the opposite phase as the keyframe.

    Weight is used to ensure that we can be certain that the new labels are correct. A
    higher weight means we will look further into the "unknown", further out towards the
    previous or next maximum difference from a keyframe.
    """
    # Blur difference a bit to make it more robust
    ed_i_diff = gaussian_filter1d(np.sum(np.abs(video - video[ed_i]), axis=(1, 2)), 5, mode="nearest")
    es_i_diff = gaussian_filter1d(np.sum(np.abs(video - video[es_i]), axis=(1, 2)), 5, mode="nearest")

    # Either ED is labeled first, or ES is. The code logic is the same, but different
    # labels have to be returned -- i.e.: it's almost copy-paste in the two clauses below.
    if ed_i < es_i:
        some_before_ed_i = int(previous_peak(ed_i_diff, ed_i) * p + ed_i * (1 - p))
        some_after_es_i = int(next_peak(es_i_diff, es_i) * p + es_i * (1 - p))
        ground_truth = (
            [0] * (ed_i - some_before_ed_i + 1)  # Diastole
            + [1] * (es_i - ed_i)  # Systole
            + [0] * (some_after_es_i - es_i)  # Diastole
        )
        return ground_truth, some_before_ed_i, some_after_es_i
    else:
        some_before_es_i = int(previous_peak(es_i_diff, es_i) * p + es_i * (1 - p))
        some_after_ed_i = int(next_peak(ed_i_diff, ed_i) * p + ed_i * (1 - p))
        ground_truth = (
            [1] * (es_i - some_before_es_i + 1)  # Systole
            + [0] * (ed_i - es_i)  # Diastole
            + [1] * (some_after_ed_i - ed_i)  # Systole
        )
        return ground_truth, some_before_es_i, some_after_ed_i
