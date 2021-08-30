import os
import random
from concurrent.futures.thread import ThreadPoolExecutor

import cv2
import edesdetectrl.util.generators as generators
import numpy as np
import pandas as pd
from scipy.ndimage.filters import gaussian_filter


def loadvideo(filename: str) -> np.ndarray:
    # Copied and modified from https://github.com/echonet/dynamic/blob/f6d9b342ffd6e5129bcf15a994e1734fc31003f6/echonet/utils/__init__.py#L16
    """Loads a video from a file.
    Args:
        filename (str): filename of video
    Returns:
        A np.ndarray with dimensions (frames, height, width). The values
        will be uint8's ranging from 0 to 255.
    Raises:
        FileNotFoundError: Could not find `filename`
        ValueError: An error occurred while reading the video
    """

    if not os.path.exists(filename):
        raise FileNotFoundError(filename)
    capture = cv2.VideoCapture(filename)

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    v = np.zeros((frame_count, frame_height, frame_width), np.uint8)

    for count in range(frame_count):
        ret, frame = capture.read()
        if not ret:
            raise ValueError("Failed to load frame #{} of {}.".format(count, filename))

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Use grayscale images
        v[count, :, :] = frame

    return v


def next_maximum_diff(x, start_i):
    increasing = np.gradient(gaussian_filter(x, sigma=2)) < 0

    max_diff_i = start_i
    for i in range(start_i + 1, len(x)):
        if increasing[i]:
            break
        else:
            max_diff_i = i

    return max_diff_i


def prev_maximum_diff(x, start_i):
    increasing = np.gradient(gaussian_filter(x, sigma=2)) >= 0

    max_diff_i = start_i
    for i in range(start_i - 1, 0, -1):
        if increasing[i]:
            break
        else:
            max_diff_i = i

    return max_diff_i


# TODO: Add test(s) for this function
def label_frames(x, ed_i, es_i, weight=0.75):
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
    ed_i_diff = [np.sum((x[i] - x[ed_i]) ** 2) for i in range(x.shape[0])]
    es_i_diff = [np.sum((x[i] - x[es_i]) ** 2) for i in range(x.shape[0])]

    # Either ED is labeled first, or ES is. The code logic is the same, but different
    # labels have to be returned -- i.e.: it's almost copy-paste in the two clauses below.
    if ed_i < es_i:
        some_before_ed_i = int(
            prev_maximum_diff(ed_i_diff, ed_i) * weight + ed_i * (1 - weight)
        )
        some_after_es_i = int(
            next_maximum_diff(es_i_diff, es_i) * weight + es_i * (1 - weight)
        )
        frames = np.arange(some_before_ed_i, some_after_es_i + 1)
        labels = (
            [0] * (ed_i - some_before_ed_i + 1)  # Diastole
            + [1] * (es_i - ed_i)  # Systole
            + [0] * (some_after_es_i - es_i)  # Diastole
        )
        return (frames, labels)
    else:
        some_before_es_i = int(
            prev_maximum_diff(es_i_diff, es_i) * weight + es_i * (1 - weight)
        )
        some_after_ed_i = int(
            next_maximum_diff(ed_i_diff, ed_i) * weight + ed_i * (1 - weight)
        )
        frames = np.arange(some_before_es_i, some_after_ed_i + 1)
        labels = (
            [1] * (es_i - some_before_es_i + 1)  # Systole
            + [0] * (ed_i - es_i)  # Diastole
            + [1] * (some_after_ed_i - ed_i)  # Systole
        )
        return (frames, labels)


def get_item(filename, traces, videos_dir):
    # Traces are sorted by cross-sectional area (reference: https://github.com/echonet/dynamic/blob/master/echonet/datasets/echo.py#L213)
    # Largest (diastolic) frame is first
    ed = int(traces.iloc[0]["Frame"])
    # Smallest (systolic) frame is last
    es = int(traces.iloc[-1]["Frame"])

    video = loadvideo(videos_dir + filename)
    # Labels are either 0 or 1.
    # 0 means diastole and 1 means systole.
    frames, labels = label_frames(video, ed, es)
    video = video[frames]

    # Cut the video length down and start from a random point in time.
    # This is such that the agent won't learn that diastole always comes first,
    # which I think it does in the training data.
    # video_length_fraction = 1 / 3
    # num_frames = len(labels)
    # actual_num_frames = int(num_frames * video_length_fraction)
    # random_start_t = random.randint(0, num_frames - actual_num_frames - 1)
    #
    # video, labels = (
    #    video[random_start_t : random_start_t + actual_num_frames],
    #    labels[random_start_t : random_start_t + actual_num_frames],
    # )

    return video, labels


def ensure_file_extension(filename):
    if os.path.splitext(filename)[1] == "":
        filename = filename + ".avi"
    return filename


def get_filenames(filelist_csv_file, split=None):
    filelist_df = pd.read_csv(filelist_csv_file)
    if split:
        filelist_df = filelist_df[filelist_df["Split"] == split]
    return [ensure_file_extension(filename) for filename in filelist_df["FileName"]]


def get_generator(
    thread_pool_executor,
    volumetracings_csv_file,
    filelist_csv_file,
    videos_dir,
    split,
    buffer_maxsize=10,
):
    volumetracings_df = pd.read_csv(volumetracings_csv_file, index_col="FileName")
    filenames = get_filenames(filelist_csv_file, split)

    def task_gen():
        while True:
            filename = random.choice(filenames)
            try:
                # This may traise a KeyError when the file can not be found in the volumetracings!
                traces = volumetracings_df.loc[filename]
                yield lambda: get_item(filename, traces, videos_dir)
            except KeyError:
                continue  # Try another random filename.

    # Let's optimize the code a bit by making it multi-threaded and storing videos and labels in a buffer, ready for use.
    return generators.async_buffered(thread_pool_executor, buffer_maxsize, task_gen())


def example():
    from edesdetectrl.config import config

    volumetracings_csv_file = config["data"]["volumetracings_path"]
    filelist_csv_file = config["data"]["filelist_path"]
    videos_dir = config["data"]["videos_path"]
    split = "TRAIN"

    with ThreadPoolExecutor(max_workers=2) as thread_pool_executor:
        gen = get_generator(
            thread_pool_executor,
            volumetracings_csv_file,
            filelist_csv_file,
            videos_dir,
            split,
        )

        for _ in range(5):
            video, labels = next(gen)
