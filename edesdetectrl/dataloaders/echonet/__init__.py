import os
from typing import List, Literal, Optional, Union

import edesdetectrl.dataloaders.echonet.label_frames as lf
import numpy as np
import pandas as pd
from edesdetectrl import dataloaders
from edesdetectrl.config import config
from edesdetectrl.dataloaders.common import loadvideo


def _ensure_file_extension(filename):
    if os.path.splitext(filename)[1] == "":
        filename = filename + ".avi"
    return filename


def _has_bad_labels(filename):
    # These files must be filtered out.
    # They have been pre-filtered in `notebooks/plots/echonet.ipynb`.
    return filename in [
        "0X2507255D8DC30B4E",
        "0X280B7441A7E287B2",
        "0X2A55659AE64722AA",
        "0X31B3F20AC08B5491",
        "0X36BD2518C9D15985",
        "0X4154F112065C857B",
        "0X43DE853BD6E0C849",
        "0X49EDDBE4F26EB4E9",
        "0X4B3A70F6BD40224B",
        "0X4BBA9C8FB485C9AB",
        "0X5648DF28AE0A879F",
        "0X5789708D8B711CE9",
        "0X62120814160BA377",
        "0X67E8F2D130F1A55",
        "0X6AFD6474F2A92942",
        "0X6E02E0F24F63EFD7",
        "0X6E5824E76BEB3ECA",
        "0X798EA367FD9FD1EA",
        "0XAF14E70264D4B68",
    ]


def _get_filenames(filelist_csv_file, split=None):
    filelist_df = pd.read_csv(filelist_csv_file)
    filelist_df = filelist_df[filelist_df["FPS"] == 50]  # Filter out FPS != 50
    if split:
        filelist_df = filelist_df[filelist_df["Split"] == split]
    return [
        _ensure_file_extension(filename)
        for filename in filelist_df["FileName"]
        # Filter out mislabeled videos
        if not _has_bad_labels(filename)
    ]


def _get_item_impl(filename: str, tracing: pd.DataFrame) -> dataloaders.DataItem:
    video = loadvideo(filename)
    # Normalize pixel intensities (smallest always 0, biggest always 1)
    video = video - video.min()
    video = video / video.max()
    video = video.astype(np.float32)

    # Calculate volume "proxies", i.e., not the actual volumes, but something that we
    # can use to compare volumes to find the biggest one.
    volume_proxies = []
    for frame, lines in tracing.groupby("Frame"):
        x1, y1, x2, y2 = lines["X1"], lines["Y1"], lines["X2"], lines["Y2"]
        volume_proxy = np.sum(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
        volume_proxies.append((frame, volume_proxy))
    ed_frame = max(volume_proxies, key=lambda x: x[1])[0]
    es_frame = min(volume_proxies, key=lambda x: x[1])[0]

    # Label the frames of the video given the ED and ES frame
    ground_truth, start, end = lf.label_frames(video, ed_frame, es_frame)

    return dataloaders.DataItem.from_video_and_ground_truth(
        filename, video, ground_truth, start, end
    )


class Echonet(dataloaders.DataLoader):
    def __init__(self, split: Optional[Literal["TRAIN", "VAL", "TEST"]]):
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

        task_fn = _get_item_impl
        args = (self.videos_dir + filename, traces)
        return task_fn, args
