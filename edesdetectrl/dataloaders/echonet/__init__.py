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
        "0X131EE16CE9512E4E",
        "0X13D1459C51B5C32E",
        "0X1430480BD524AA15",
        "0X1597DA4581B6F4AD",
        "0X163A374094D066E8",
        "0X1CB1B34B202C4041",
        "0X1F7896620951FFD",
        "0X20FC953931845AED",
        "0X23FCF45E798060D0",
        "0X2507255D8DC30B4E",
        "0X280B7441A7E287B2",
        "0X29DC850F81AD37D8",
        "0X2D93EB042A3DB0E2",
        "0X2DAE446A22A6038",
        "0X2EC2E75D27C6F224",
        "0X303781B228664E4",
        "0X31B3F20AC08B5491",
        "0X352D7150FCBFA7C1",
        "0X357D90B1EE59CC94",
        "0X35A5E9C9075E56EE",
        "0X366AD377E4A81FBE",
        "0X367085DDC2E90657",
        "0X3902A366CE2F62E8",
        "0X3A84F1E3BCC9B6E6",
        "0X3BA9F7C9DB0CF55B",
        "0X3D1241A509AEDD5D",
        "0X3D8EE56C7B00E120",
        "0X3EB0FC2695B0AB5F",
        "0X4154F112065C857B",
        "0X43DE853BD6E0C849",
        "0X46ACC9C2CF9CFB1E",
        "0X49EDDBE4F26EB4E9",
        "0X4B3A70F6BD40224B",
        "0X4BBA9C8FB485C9AB",
        "0X4E9496513479C330",
        "0X4E9F08061D109568",
        "0X4F5499DDFF536F69",
        "0X5042C6AB36212224",
        "0X5083D1646DF2023E",
        "0X511C9798FB92301C",
        "0X51F3A05DB9647C01",
        "0X52E865A8C311F559",
        "0X551E297304F24B60",
        "0X575A1E4C8C441849",
        "0X5789708D8B711CE9",
        "0X592F522398D46067",
        "0X59D5F41F45601E03",
        "0X5E9CA0D95225A239",
        "0X5FD48AFE4017BCC",
        "0X5FE6439A0CCEF482",
        "0X6210992ADCECE486",
        "0X63A862F83C131D52",
        "0X642E639A8CDE539B",
        "0X67E8F2D130F1A55",
        "0X67F8AC58B0BAA98",
        "0X6CCA2353460A1836",
        "0X6D872CF7C744613E",
        "0X6DC4325C689CEEDD",
        "0X6E02E0F24F63EFD7",
        "0X6E5824E76BEB3ECA",
        "0X725C7955468D7BAA",
        "0X72B134B9DB954CE0",
        "0X7361844FD9363EEA",
        "0X759B7B28B1086707",
        "0X766B7B0ABDB07CD5",
        "0X778D8A645214CBB0",
        "0X781152F1005A1522",
        "0X7925C344B4A5872F",
        "0X7A0E6F5825AEC419",
        "0X7B9A154FC4B9A975",
        "0X7F33CFDB31ADCD6A",
        "0X803C3B563A14E1F",
        "0X9D0FC1FDA17491D",
        "0XAA3A306D4EC8B69",
        "0XB8513240ED5E94",
        "0XBC881D0528B2788",
        "0XD5F2509FE1D0C20",
        "0XE09693AED1032C0",
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

        # Traces are sorted by cross-sectional area (reference: https://github.com/echonet/dynamic/blob/master/echonet/datasets/echo.py#L213)
        # Largest (diastolic) frame is first, smallest (systolic) frame is last
        ed, es = int(traces.iloc[0]["Frame"]), int(traces.iloc[-1]["Frame"])

        task_fn = _get_item_impl
        args = (self.videos_dir + filename, ed, es)
        return task_fn, args
