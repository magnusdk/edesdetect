import os
from typing import Any, Callable, List, Literal, Tuple, Type, Union

import cv2
import numpy as np
import pandas as pd
from edesdetectrl import dataloaders
from edesdetectrl.config import config
import edesdetectrl.dataloaders.echonet.label_frames as lf


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


def _ensure_file_extension(filename):
    if os.path.splitext(filename)[1] == "":
        filename = filename + ".avi"
    return filename


def _get_filenames(filelist_csv_file, split=None):
    filelist_df = pd.read_csv(filelist_csv_file)
    if split:
        filelist_df = filelist_df[filelist_df["Split"] == split]
    return [_ensure_file_extension(filename) for filename in filelist_df["FileName"]]


def _get_item_impl(filename: str, ed_frame: int, es_frame: int) -> dataloaders.DataItem:
    video = loadvideo(filename)
    # Normalize pixel intensities (smallest always 0, biggest always 1)
    video = video - video.min()
    video = video / video.max()
    video = video.astype(np.float32)

    ground_truth, start, end = lf.label_frames(video, ed_frame, es_frame)

    return dataloaders.DataItem.from_video_and_ground_truth(
        filename, video, ground_truth, start, end
    )


class Echonet(dataloaders.DataLoader):
    def __init__(self, split: Literal["TRAIN", "VAL", "TEST"]):
        self.filelist_csv_file = config["data"]["filelist_path"]
        self.volumetracings_df = pd.read_csv(
            config["data"]["volumetracings_path"], index_col="FileName"
        )
        self.videos_dir = config["data"]["videos_path"]

        self.filenames = _get_filenames(self.filelist_csv_file, split)

    @property
    def keys(self) -> List[str]:
        return self.filenames

    def __getitem__(self, key: Union[str, int]) -> dataloaders.DataloaderTask:
        filename = key if isinstance(key, str) else self.keys[key]
        traces = self.volumetracings_df.loc[filename]

        # Traces are sorted by cross-sectional area (reference: https://github.com/echonet/dynamic/blob/master/echonet/datasets/echo.py#L213)
        # Largest (diastolic) frame is first, smallest (systolic) frame is last
        ed, es = int(traces.iloc[0]["Frame"]), int(traces.iloc[-1]["Frame"])

        task_fn = _get_item_impl
        args = (self.videos_dir + filename, ed, es)
        return task_fn, args