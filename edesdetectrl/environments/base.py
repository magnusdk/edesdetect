from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Literal, Sequence, Tuple, Union

import gym
import numpy as np
from edesdetectrl import dataloaders
from edesdetectrl.util.generators import async_buffered

ObservationFn = Callable[["BinaryClassificationBaseEnv"], Any]
Action = int
RewardFn = Callable[["BinaryClassificationBaseEnv", Action], float]


class BinaryClassificationBaseEnv(gym.Env):
    def __init__(
        self,
        pad_left: int,
        pad_right: int,
        get_observation: ObservationFn,
        get_reward: RewardFn,
    ):
        super(BinaryClassificationBaseEnv, self).__init__()
        self.pad_left = pad_left
        self.pad_right = pad_right
        self.get_reward = get_reward
        self.get_observation = get_observation
        self._video, self._ground_truth = None, None

    def is_ready(self):
        return self._video is not None and self._ground_truth is not None

    def reset(self):
        assert self.is_ready(), "Video and ground_truth must be set."
        self.current_frame = self.pad_left
        observation = self.get_observation(self)
        return observation

    def step(self, action):
        assert self.is_ready(), "Video and ground_truth must be set."

        reward = self.get_reward(self, action)
        info = {"ground_truth": self._ground_truth[self.current_frame]}

        # Go to the next frame if the action was Diastole or Systole
        # and if there still are enough padding on the right to go to the next frame
        enough_padding = self.current_frame + self.pad_right < self._video.shape[0] - 1
        if action in (0, 1) and enough_padding:
            self.current_frame += 1
        done = self.current_frame + self.pad_right >= self._video.shape[0] - 1
        observation = self.get_observation(self)

        return observation, reward, done, info

    @property
    def video_and_labels(self):
        return self._video, self._ground_truth

    @video_and_labels.setter
    def video_and_labels(self, new_value: Tuple[np.ndarray, Sequence[int]]):
        video, ground_truth = new_value
        n_frames = video.shape[0]
        assert (
            ground_truth is not None and len(ground_truth) == n_frames
        ), "Ground truth must have the same number of labels as there are frames in the seq."
        assert (
            video is not None and n_frames >= self.pad_left + self.pad_right + 1
        ), f"Video must have atleast {self.pad_left+self.pad_right+1} frames."
        self._video, self._ground_truth = video, ground_truth


class DataIteratorMixin:
    """Add data iterator and set video_and_labels."""

    def __init__(
        self,
        dataloader: Union[dataloaders.DataLoader, Literal["echonet"]],
        min_video_length: int,
    ):
        if dataloader == "echonet":
            from edesdetectrl.dataloaders.echonet import Echonet

            dataloader = Echonet("TRAIN")

        self.data_iterator = filter(
            lambda v: v[0].shape[0] >= min_video_length,
            # TODO: async_buffered help much after all. Find better ways to speed things up.
            async_buffered(
                dataloader.get_generator(),
                ThreadPoolExecutor(),
                5,
            ),
        )
        self.video_and_labels = next(self.data_iterator)

    def next_video_and_labels(self):
        self.video_and_labels = next(self.data_iterator)
