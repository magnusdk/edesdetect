import math
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Union

import cv2
import numpy as np
import pandas as pd
from edesdetectrl import dataloaders
from edesdetectrl.config import config
from edesdetectrl.dataloaders import DataItem, GroundTruth
from edesdetectrl.dataloaders.common import loadvideo


@dataclass
class EchoTimingDataItem:
    avi_name: str
    number_of_frames: int
    operator_number: int
    ed_events: List[int]
    es_events: List[int]


def get_beats(
    row: pd.Series,
    operator_number: int,
    event: Literal["ed", "es"],
) -> List[int]:
    beats = [row[f"beat{i}_{event}_op{operator_number}"] for i in range(1, 5)]
    return sorted([int(beat) for beat in beats if not math.isnan(beat)])


def get_ground_truth_labels(data_item: EchoTimingDataItem) -> List[GroundTruth]:
    ed_events = data_item.ed_events.copy()
    es_events = data_item.es_events.copy()
    labels = []
    while ed_events or es_events:
        if not es_events or (ed_events and ed_events[0] < es_events[0]):
            labels += ["ed"]
            ed_i = ed_events.pop(0)
            if es_events:
                labels += ["es"] * (es_events[0] - ed_i - 1)
        else:
            labels += ["es"]
            es_i = es_events.pop(0)
            if ed_events:
                labels += ["ed"] * (ed_events[0] - es_i - 1)
    return [0 if label == "ed" else 1 for label in labels]


def get_item_impl(echotiming_data_item: EchoTimingDataItem) -> DataItem:
    all_events = echotiming_data_item.es_events + echotiming_data_item.ed_events
    first_event = min(all_events)
    last_event = max(all_events)
    raw_video = loadvideo(
        config["data"]["echotiming_dir"]
        + "Videos/"
        + echotiming_data_item.avi_name
        + ".avi"
    )
    size = 112
    raw_video = raw_video[first_event : last_event + 1]
    video = np.empty((raw_video.shape[0],) + (size, size))
    for i, frame in enumerate(raw_video):
        video[i] = cv2.resize(frame, dsize=(size, size), interpolation=cv2.INTER_CUBIC)
    ground_truth_labels = get_ground_truth_labels(echotiming_data_item)
    return DataItem(
        name=echotiming_data_item.avi_name,
        video=video,
        total_length=echotiming_data_item.number_of_frames,
        ground_truth=ground_truth_labels,
        ground_truth_start=first_event,
        ground_truth_end=last_event,
        length=len(video),
        extra_frames_left=first_event,
        extra_frames_right=(echotiming_data_item.number_of_frames - last_event - 1),
    )


class EchoTiming(dataloaders.DataLoader):
    def __init__(
        self,
        split: Optional[Literal["TRAIN", "VAL", "TEST"]],
        operator_number: int = 1,
    ):
        self.split = split
        self.operator_number = operator_number

        header_names = ["no", "avi_name", "number_of_frames", "split"]
        header_names += [  # Headers for the different beat labels
            f"{beat}_{event}_{operator}"
            for beat in ["beat1", "beat2", "beat3", "beat4"]
            for event in ["ed", "es"]
            for operator in ["op1", "op2"]
        ]
        timings_df = pd.read_excel(
            config["data"]["echotiming_dir"] + "FileList_Timing.xlsx",
            names=header_names,
            skiprows=[0, 1],
        )

        self.op1: Dict[str, EchoTimingDataItem] = {}
        self.op2: Dict[str, EchoTimingDataItem] = {}
        for _, row in timings_df.iterrows():
            self.op1[row["avi_name"]] = EchoTimingDataItem(
                avi_name=row["avi_name"],
                number_of_frames=row["number_of_frames"],
                operator_number=1,
                ed_events=get_beats(row, 1, "ed"),
                es_events=get_beats(row, 1, "es"),
            )

            self.op2[row["avi_name"]] = EchoTimingDataItem(
                avi_name=row["avi_name"],
                number_of_frames=row["number_of_frames"],
                operator_number=2,
                ed_events=get_beats(row, 2, "ed"),
                es_events=get_beats(row, 2, "es"),
            )

    @property
    def keys(self) -> List[str]:
        data_items = self.op1 if self.operator_number == 1 else self.op2
        avi_names = [
            item.avi_name
            for item in data_items.values()
            # Must have at least one of each event in order to get binary labels
            # (This is True for all videos, but I keep it here to make it explicit)
            if (len(item.ed_events) >= 1 and len(item.es_events) >= 1)
        ]
        # Sort to ensure same behavior on all machines (order of values in dict is not
        # defined)
        avi_names.sort()

        # Filter split on "TRAIN" / "VAL" / "TEST"
        rng = np.random.default_rng(1995)
        rng.shuffle(avi_names)
        # Filtering is easy when the number is divisible by 5 (60% + 20% + 20%)
        assert len(avi_names) == 1000
        if self.split == "TRAIN":
            avi_names = avi_names[:600]
        elif self.split == "VAL":
            avi_names = avi_names[600:800]
        elif self.split == "TEST":
            avi_names = avi_names[800:]
        # else: return all avi_names

        return avi_names

    def __getitem__(self, key: Union[str, int]) -> dataloaders.DataloaderTask:
        filename = key if isinstance(key, str) else self.keys[key]
        operator_data = self.op1 if self.operator_number == 1 else self.op2
        echonet_timing_data_item = operator_data[filename]
        return get_item_impl, (echonet_timing_data_item,)


# TODO: Code stuff
# - Filter out anything other than the sector scan itself
# - Normalize FPS maybe?
# - Filter on color and b
#
#
#
# TODO: Write in report:
# - Need at least one ED and one ES to make ground truth labels. This is true for all 1000 videos.
# - Split the dataset in train/val/test like in Elizabeth Lane (if possible)
#
#
#
# TODO: Experiment with:
# - Using 300x300 videos instead of 112x112
#
#
