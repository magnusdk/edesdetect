# This namespace contains a re-implementation of acme.wrappers.GymWrapper
# that also stores the info returned from a gym.Env object step function.

import gym
from acme import types, wrappers

import dm_env
from typing import Any, NamedTuple


class TimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    info: Any

    def first(self) -> bool:
        return self.step_type == dm_env.StepType.FIRST

    def mid(self) -> bool:
        return self.step_type == dm_env.StepType.MID

    def last(self) -> bool:
        return self.step_type == dm_env.StepType.LAST


def restart(observation):
    """Returns a `TimeStep` with `step_type` set to `StepType.FIRST`."""
    return TimeStep(dm_env.StepType.FIRST, None, None, observation, None)


def transition(reward, observation, info, discount=1.0):
    """Returns a `TimeStep` with `step_type` set to `StepType.MID`."""
    return TimeStep(dm_env.StepType.MID, reward, discount, observation, info)


def termination(reward, observation, info):
    """Returns a `TimeStep` with `step_type` set to `StepType.LAST`."""
    return TimeStep(dm_env.StepType.LAST, reward, 0.0, observation, info)


def truncation(reward, observation, info, discount=1.0):
    """Returns a `TimeStep` with `step_type` set to `StepType.LAST`."""
    return TimeStep(dm_env.StepType.LAST, reward, discount, observation, info)


class GymWrapper(wrappers.GymWrapper):
    def __init__(self, environment: gym.Env):
        super().__init__(environment)

    def step(self, action: types.NestedArray) -> TimeStep:
        """Steps the environment."""
        if self._reset_next_step:
            return self.reset()

        observation, reward, done, info = self._environment.step(action)
        self._reset_next_step = done

        if done:
            truncated = info.get("TimeLimit.truncated", False)
            if truncated:
                return truncation(reward, observation, info)
            return termination(reward, observation, info)
        return transition(reward, observation, info)