import itertools
from dataclasses import dataclass
from itertools import count
from typing import Any, Callable, Generator, Literal, Sequence, Tuple, Union

import edesdetectrl.util.generators as generators
import numpy as np
from edesdetectrl.util.concurrent_pool import process_pool
from jax import random
from jax._src.random import KeyArray

GroundTruth = Literal[0, 1]


@dataclass
class DataItem:
    name: str  # Name/ID of the video

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
        name: str,
        video: np.ndarray,
        ground_truth: Sequence[GroundTruth],
        ground_truth_start: int,
        ground_truth_end: int,
    ) -> "DataItem":
        return DataItem(
            name=name,
            video=video,
            total_length=len(video),
            ground_truth=ground_truth,
            ground_truth_start=ground_truth_start,
            ground_truth_end=ground_truth_end,
            length=len(ground_truth),
            extra_frames_left=ground_truth_start,
            extra_frames_right=len(video) - ground_truth_end,
        )


# A tuple of a function and args. The function can be called with the args.
DataloaderTask = Tuple[Callable[..., DataItem], Tuple]


def _safe_get_item(items: "DataLoader", index: int) -> Union[DataloaderTask, str]:
    try:
        return items[index]
    except KeyError:
        return generators.SKIP_ITEM


class DataLoader:
    @property
    def keys(self):
        raise NotImplementedError

    def __getitem__(self, index: Any) -> DataloaderTask:
        """Return a DataloaderTask which is a tuple of task_fn and args.
        We can call task_fn with args as such: task_fn(*args), and it will return a DataItem."""
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.keys)

    def get_random_generator(
        self, rng_key: KeyArray
    ) -> Generator[DataItem, None, None]:
        """Return a generator that generates all items from dataloader in random order."""

        def task_gen():
            key1 = rng_key
            while True:
                key1, key2 = random.split(key1)
                index = random.randint(key2, (1,), 0, len(self))[0]
                yield _safe_get_item(self, index)

        return generators.async_buffered(task_gen(), process_pool, 20)

    def get_generator(self, cycle=True, prefetch=20) -> Generator[DataItem, None, None]:
        """Return a generator that generates all items from dataloader, ordered by index.

        If cycle is True, then the generator will generate indefinitely, cycling back to
        the start after all items have been generated."""

        def task_gen():
            for n in count():
                index = n % len(self)
                yield _safe_get_item(self, index)

        gen = generators.async_buffered(task_gen(), process_pool, prefetch)
        # Stop once the last dataitem has been generated if cycle is False
        if not cycle:
            gen = itertools.islice(gen, len(self))

        return gen
