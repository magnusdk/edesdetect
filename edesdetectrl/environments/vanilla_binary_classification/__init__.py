from typing import Literal, Optional, Union

import gym
import numpy as np
from edesdetectrl import dataloaders
from edesdetectrl.environments import rewards
from edesdetectrl.environments.base import (
    BinaryClassificationBaseEnv,
    DataIteratorMixin,
    RewardFn,
)
from edesdetectrl.environments.vanilla_binary_classification import render
from gym import spaces
from jax._src.random import KeyArray

actions = {
    "DIASTOLE": 0,
    "SYSTOLE": 1,
}
# Action name lookup indexed by action integer
iactions = {v: k for k, v in actions.items()}
# Let's make it more robust by also supporting indexing by string
iactions = {**iactions, **{str(v): k for k, v in actions.items()}}


N_PADDING = 3
N_CHANNELS = 2 * N_PADDING + 1
N_ACTIONS = len(actions)
HEIGHT = 112
WIDTH = 112


def get_observation(env: BinaryClassificationBaseEnv):
    frame = env.current_frame
    video = env._video
    observation = video[frame - N_PADDING : frame + N_PADDING + 1]
    return observation.astype("float32")


class VanillaBinaryClassificationBase_v0(BinaryClassificationBaseEnv):
    metadata = {"render.modes": ["rgb_array"], "actions": actions}

    def __init__(
        self,
        get_reward: Union[RewardFn, Literal["simple", "proximity"]],
    ):
        get_reward = (
            rewards.simple_reward
            if get_reward == "simple"
            else rewards.proximity_reward
            if get_reward == "proximity"
            else get_reward
        )
        super().__init__(N_PADDING, N_PADDING, get_observation, get_reward)

        self.action_space = spaces.Discrete(N_ACTIONS)
        self.action_space.dtype = np.int32
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(N_CHANNELS, HEIGHT, WIDTH), dtype="float32"
        )

    def step(self, action):
        if isinstance(action, str):
            # Sanitize action value by converting to action_name and back to the canonical representation (integer)
            action = actions[iactions[action]]
        return BinaryClassificationBaseEnv.step(self, action)

    def render(self, mode="rgb_array"):
        assert mode == "rgb_array"
        observation = self.get_observation(self)
        return render.render_observation(observation)


class VanillaBinaryClassification_v0(
    DataIteratorMixin, VanillaBinaryClassificationBase_v0
):
    def __init__(
        self,
        dataloader: Union[dataloaders.DataLoader, Literal["echonet"]],
        get_reward: Union[RewardFn, Literal["simple", "proximity"]],
        rng_key: Optional[KeyArray] = None,
    ):
        VanillaBinaryClassificationBase_v0.__init__(self, get_reward)
        DataIteratorMixin.__init__(self, dataloader, N_CHANNELS, rng_key)

    def reset(self):
        self.next_video_and_labels()
        return VanillaBinaryClassificationBase_v0.reset(self)


gym.register(
    id="VanillaBinaryClassification-v0",
    entry_point="edesdetectrl.environments.vanilla_binary_classification:VanillaBinaryClassification_v0",
    max_episode_steps=200,
)
