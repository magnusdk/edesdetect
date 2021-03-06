from collections import deque
from typing import Literal, Optional, Tuple, Union

import gym
import numpy as np
from edesdetectrl import dataloaders
from edesdetectrl.environments import rewards
from edesdetectrl.environments.base import (
    BinaryClassificationBaseEnv,
    DataIteratorMixin,
    RewardFn,
)
from edesdetectrl.environments.m_mode_binary_classification.line import MModeLine
from edesdetectrl.environments.m_mode_binary_classification.render import (
    render_observation,
)
from gym import spaces
from jax import random
from jax._src.random import KeyArray

actions = {
    "DIASTOLE": 0,
    "SYSTOLE": 1,
    "UP": 2,
    "DOWN": 3,
    "LEFT": 4,
    "RIGHT": 5,
    "ROTATE_LEFT": 6,
    "ROTATE_RIGHT": 7,
}
# Action name lookup indexed by action integer
iactions = {v: k for k, v in actions.items()}
# Let's make it more robust by also supporting indexing by string
iactions = {**iactions, **{str(v): k for k, v in actions.items()}}

WIDTH = 112
HEIGHT = 112
N_OVERVIEW_CHANNELS = 2
N_MMODE_CHANNELS = 9
N_ACTIONS = len(actions)
N_PADDING = 15
N_FRAMES = N_PADDING * 2 + 1
LINE_LENGTH = 64
STEP_SIZE = 1
ROTATION_AMOUNT = 0.1


def get_mmode_observation(env: "EDESMModeClassificationBase_v0") -> np.ndarray:
    # M-mode data
    video_start = env.current_frame - N_PADDING
    video_end = env.current_frame + N_PADDING + 1
    video = env.video.video[video_start:video_end]

    rotations = [-0.5, 0, 0.5]
    horizontal_translations = [-5, 0, 5]
    mmode_with_adjacent = env.mmode_line.get_mmode_image_with_adjacent(
        video,
        rotations=rotations,
        horizontal_translations=horizontal_translations,
    )
    # The main M-mode image is the one that has no rotation and no translation
    # Since we have an equal number of adjacent on either side (i.e. f.ex. the middle rotation is 0) then the main M-mode image is the middle adjacent.
    # Let's assert this so we don't make mistakes in the future.
    assert rotations[len(rotations) // 2] == 0
    assert horizontal_translations[len(horizontal_translations) // 2] == 0
    main_mmode_index = (len(rotations) * len(horizontal_translations)) // 2
    main_mmode_image = mmode_with_adjacent[main_mmode_index]
    # Use the difference from the main M-mode image for adjacent images
    mmode_with_adjacent = mmode_with_adjacent - main_mmode_image
    mmode_with_adjacent[main_mmode_index] = main_mmode_image

    return mmode_with_adjacent


def get_action_history_observation(env: "EDESMModeClassificationBase_v0") -> np.ndarray:
    a_hist_array = np.zeros((env.action_history.maxlen, 8), dtype="float32")
    for i, action in enumerate(env.action_history):
        # action is an int. We don't include the first two actions (diastole and
        # systole) so we subtract those indices.
        a_hist_array[i, action - 2] = 1.0
    return a_hist_array


def get_overview_and_mmode_observation(
    env: "EDESMModeClassificationBase_v0",
) -> Tuple[np.ndarray, np.ndarray]:
    """Return a tuple of 2 arrays:
    1. overview_data: An array with 2 channels that is the average pixel intensities in the video and the location of the line.
    2. mmode_data: An array with 9 channels that is the M-mode image of the current line and the 8 adjacent lines."""
    # Overview data
    mean_image = np.mean(
        env.video.video[:50], axis=0
    )  # Average of first 50 frames in video
    line_pixels = env.mmode_line.visualize_line()
    overview_data = np.array([mean_image, line_pixels])
    overview_data = np.transpose(overview_data, (1, 2, 0))

    # M-mode data
    mmode_with_adjacent = get_mmode_observation(env)
    mmode_with_adjacent = np.transpose(mmode_with_adjacent, (1, 2, 0))

    # Action history data
    action_history = get_action_history_observation(env)

    return overview_data, mmode_with_adjacent, action_history


class EDESMModeClassificationBase_v0(BinaryClassificationBaseEnv):
    metadata = {"render.modes": ["rgb_array"], "actions": actions}

    def __init__(
        self,
        get_reward: Union[RewardFn, Literal["simple", "proximity"]],
        rng_key: Union[int, KeyArray],
    ):
        get_reward = (
            rewards.simple_reward
            if get_reward == "simple"
            else rewards.proximity_reward
            if get_reward == "proximity"
            else get_reward
        )
        rng_key = random.PRNGKey(rng_key) if isinstance(rng_key, int) else rng_key
        super().__init__(
            N_PADDING, N_PADDING, get_overview_and_mmode_observation, get_reward
        )
        self.rng_key = rng_key
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.action_space.dtype = np.int32

        # Observations are tuples of:
        # 1. an averaged image of the video and the current line (2 channels)
        # 2. and the M-mode image with adjacent lines (1+4 channels for 4 adjacent)
        overview_space = spaces.Box(
            low=0, high=1, shape=(HEIGHT, WIDTH, N_OVERVIEW_CHANNELS), dtype="float32"
        )
        m_mode_space = spaces.Box(
            low=0,
            high=1,
            shape=(N_FRAMES, LINE_LENGTH, N_MMODE_CHANNELS),
            dtype="float32",
        )
        action_history_space = spaces.Box(low=0, high=1, shape=(5, 8), dtype="float32")
        self.observation_space = spaces.Tuple(
            [overview_space, m_mode_space, action_history_space]
        )

        self.mmode_line = MModeLine.from_shape(
            self.rng_key, WIDTH, HEIGHT, n_line_samples=LINE_LENGTH
        )
        self.action_history = deque(maxlen=5)
        self.time_since_last_prediction = 0

        self.prev_positions = set()

    def step(self, action):
        # Sanitize action value by converting to action_name and back to the canonical representation (integer)
        action_name = iactions[action.tolist()]
        action = actions[action_name]

        old_line = self.mmode_line.line
        if action_name == "UP":
            self.mmode_line.move_vertically(-STEP_SIZE)
        elif action_name == "DOWN":
            self.mmode_line.move_vertically(STEP_SIZE)
        elif action_name == "LEFT":
            self.mmode_line.move_horizontally(-STEP_SIZE)
        elif action_name == "RIGHT":
            self.mmode_line.move_horizontally(STEP_SIZE)
        elif action_name == "ROTATE_LEFT":
            self.mmode_line.rotate(ROTATION_AMOUNT)
        elif action_name == "ROTATE_RIGHT":
            self.mmode_line.rotate(-ROTATION_AMOUNT)

        if action_name not in ("DIASTOLE", "SYSTOLE"):
            # If some kind of movement
            self.time_since_last_prediction += 1
            self.prev_positions.add(old_line)
        else:
            self.time_since_last_prediction = 0
            self.action_history.append(action)
            self.prev_positions = set()

        # Move the line to a random position and give penalty if agent moves out of bounds
        reward = None
        if old_line == self.mmode_line.line and action_name not in (
            "DIASTOLE",
            "SYSTOLE",
        ):
            reward = float(-1)
            self.mmode_line = MModeLine.from_shape(
                self.rng_key, WIDTH, HEIGHT, n_line_samples=LINE_LENGTH
            )
            (self.rng_key,) = random.split(self.rng_key, 1)
        else:
            reward = self.get_reward(self, action)
            # if action_name not in ("DIASTOLE", "SYSTOLE"):
            #    reward /= (1+self.time_since_last_prediction*0.5)

        done = False
        if self.mmode_line.line in self.prev_positions:
            reward = -1.0
            self.rng_key, sub_key = random.split(self.rng_key, 2)
            self.mmode_line = MModeLine.from_shape(
                sub_key, WIDTH, HEIGHT, n_line_samples=LINE_LENGTH
            )
        return BinaryClassificationBaseEnv.step(self, action, reward=reward, done=done)

    def reset(self):
        self.mmode_line = MModeLine.from_shape(
            self.rng_key, WIDTH, HEIGHT, n_line_samples=LINE_LENGTH
        )
        self.action_history = deque(maxlen=5)
        self.time_since_last_prediction = 0
        self.prev_positions = set()
        (self.rng_key,) = random.split(self.rng_key, 1)
        return BinaryClassificationBaseEnv.reset(self)

    def render(self, mode="rgb_array"):
        assert mode == "rgb_array"
        mean_image_data, mmode_image = get_overview_and_mmode_observation(self)
        return render_observation(mean_image_data, mmode_image)


class EDESMModeClassification_v0(DataIteratorMixin, EDESMModeClassificationBase_v0):
    def __init__(
        self,
        dataloader: Union[dataloaders.DataLoader, Literal["echonet"]],
        get_reward: Union[RewardFn, Literal["simple", "proximity"]],
        rng_key: KeyArray,
        dataloader_rng_key: Optional[KeyArray] = None,
    ):
        EDESMModeClassificationBase_v0.__init__(self, get_reward, rng_key)
        DataIteratorMixin.__init__(
            self, dataloader, N_PADDING, N_PADDING, dataloader_rng_key
        )

    def reset(self):
        self.next_video_and_labels()
        return EDESMModeClassificationBase_v0.reset(self)


gym.register(
    id="EDESMModeClassification-v0",
    entry_point="edesdetectrl.environments.m_mode_binary_classification:EDESMModeClassification_v0",
    max_episode_steps=200,
)
