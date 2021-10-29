import gym
from gym import spaces


class SlidingState(gym.Env):
    """A mock-environment that has a list of pre-defined states that are
    iterated (slided) through after each step. The goal is to pick the
    action that equals the state at a given point."""
    def __init__(self, states):
        self.s = states
        self.i = 0
        self.observation_space = spaces.Discrete(2)
        self.action_space = spaces.Discrete(2)
        super().__init__()

    def reset(self):
        self.i = 0
        return self.s[self.i]

    def step(self, action):
        ground_truth = self.s[self.i]
        r = 1 if action == ground_truth else 0
        info = {"ground_truth": ground_truth}
        
        self.i += 1
        done = self.i >= len(self.s) - 1
        return self.s[self.i], r, done, info
