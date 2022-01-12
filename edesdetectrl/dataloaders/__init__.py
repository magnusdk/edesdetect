from dataclasses import dataclass
from itertools import count
from typing import Any, Callable, Generator, Literal, Sequence, Union

import edesdetectrl.util.generators as generators
import numpy as np
from jax import random
from jax._src.random import KeyArray

GroundTruth = Literal[0, 1]


@dataclass
class DataItem:
    video: np.ndarray  # Video data
    total_length: int  # Total length of the video

    ground_truth: Sequence[GroundTruth]  # Ground truth for labelled frames
    ground_truth_start: int  # Start of labelled frames (inclusive)
    ground_truth_end: int  # Last labelled frames (inclusive)
    length: int  # Number of labelled frames

    extra_frames_left: int  # Number of extra frames before the labelled frames
    extra_frames_right: int  # Number of extra frames after the labelled frames

    @staticmethod
    def from_video_and_ground_truth(
        video: np.ndarray,
        ground_truth: Sequence[GroundTruth],
        ground_truth_start: int,
        ground_truth_end: int,
    ) -> "DataItem":
        return DataItem(
            video=video,
            total_length=len(video),
            ground_truth=ground_truth,
            ground_truth_start=ground_truth_start,
            ground_truth_end=ground_truth_end,
            length=len(ground_truth),
            extra_frames_left=ground_truth_start,
            extra_frames_right=len(video) - ground_truth_end,
        )


def _safe_get_item(items: "DataLoader", index: int) -> DataItem:
    try:
        return items[index]
    except KeyError:
        return generators.SKIP_ITEM


class DataLoader:
    @property
    def keys(self):
        raise NotImplementedError

    def __getitem__(self, index) -> DataItem:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.keys)

    def get_random_generator(
        self, rng_key: KeyArray, as_task: bool = True
    ) -> Generator[Union[DataItem, Callable[[], DataItem]], None, None]:
        """Return a generator that generates all items from dataloader in random order.

        If as_task is True, then generated values are 0-arity functions that can be
        called to get the actual value. This is useful when wrapping the generator in
        an asynchronous generator."""
        key1 = rng_key
        while True:
            key1, key2 = random.split(key1)
            task = lambda: _safe_get_item(
                self, random.randint(key2, (1,), 0, len(self))[0]
            )
            yield task if as_task else task()

    def get_generator(
        self, as_task: bool = True
    ) -> Generator[Union[DataItem, Callable[[], DataItem]], None, None]:
        """Return a generator that generates all items from dataloader, ordered by index.

        The generator will generate indefinitely, cycling back to the start after all items have been generated.

        If as_task is True, then generated values are 0-arity functions that can be
        called to get the actual value. This is useful when wrapping the generator in
        an asynchronous generator."""
        for n in count():
            index = n % len(self)
            task = lambda: _safe_get_item(self, index)
            yield task if as_task else task()
