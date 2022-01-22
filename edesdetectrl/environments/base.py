from typing import Any, Callable, Literal, Optional, Union

import gym
from edesdetectrl import dataloaders
from jax._src.random import KeyArray

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
        self._video: dataloaders.DataItem = None

    def is_ready(self):
        return self.video is not None

    def reset(self):
        assert self.is_ready(), "Video must be set."
        self.current_frame = self.video.ground_truth_start
        observation = self.get_observation(self)
        return observation

    def step(self, action):
        assert self.is_ready(), "Video must be set."

        reward = self.get_reward(self, action)
        info = {
            "ground_truth": self.video.ground_truth[
                self.current_frame - self.video.ground_truth_start
            ]
        }

        # Go to the next frame if the action was Diastole or Systole
        # and if we have not yet reached the end of labeled frames
        if action in (0, 1) and self.current_frame < self.video.ground_truth_end:
            self.current_frame += 1
        done = self.current_frame >= self.video.ground_truth_end
        observation = self.get_observation(self)

        return observation, reward, done, info

    @property
    def video(self) -> dataloaders.DataItem:
        return self._video

    @video.setter
    def video(self, video: dataloaders.DataItem):
        assert video is not None, "Video must not be None."
        assert video.length >= 1, "There must be at least one labelled frame."
        assert (
            video.extra_frames_left >= self.pad_left
        ), f"Video must have at least {self.pad_left} frames before the first labelled frame."
        assert (
            video.extra_frames_right >= self.pad_right
        ), f"Video must have at least {self.pad_right} frames after the last labelled frame."
        self._video = video


class DataIteratorMixin:
    """Add data iterator and set video_and_labels."""

    def __init__(
        self,
        dataloader: Union[dataloaders.DataLoader, Literal["echonet"]],
        pad_left: int,
        pad_right: int,
        rng_key: Optional[KeyArray] = None,
    ):
        if dataloader == "echonet":
            from edesdetectrl.dataloaders.echonet import Echonet

            dataloader = Echonet("TRAIN")

        def has_enough_frames(video: dataloaders.DataItem):
            return (
                isinstance(video, dataloaders.DataItem)
                and video.extra_frames_left >= pad_left
                and video.extra_frames_right >= pad_right
            )

        self.data_iterator = filter(
            has_enough_frames,
            dataloader.get_random_generator(rng_key)
            if rng_key is not None
            else dataloader.get_generator(),
        )
        self.video = next(self.data_iterator)

    def next_video_and_labels(self):
        self.video = next(self.data_iterator)
