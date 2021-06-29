import jax.numpy as jnp
import random
import gym
from gym import spaces
import numpy as np

from edesdetectrl.util.echonet_loader import EchoNetDataset
from edesdetectrl.util.plot_animation import plot_animation


def get_reward(current_frame, current_prediction, all_predictions, ground_truths):
    # Is the current prediction correct?
    r1 = 1 if current_prediction == ground_truths[current_frame] else 0

    return r1


def get_observation(seq, next_frame):
    if 0 < next_frame < seq.shape[0]:
        return seq[next_frame]
    else:
        return jnp.zeros_like(seq[0])


def get_seq_iterator():
    echonet_dataset = EchoNetDataset()

    while True:
        index = random.randint(0, echonet_dataset.num_videos - 1)
        video, ground_truth = echonet_dataset.get_video_and_labels(index)

        # Cut the video length down and start from a random point in time.
        # This is such that the agent won't learn that diastole always comes first,
        # which I think it does in the training data.
        video_length_fraction = 1/3
        num_frames = len(ground_truth)
        actual_num_frames = int(num_frames * video_length_fraction)
        random_start_t = random.randint(0, num_frames-actual_num_frames-1)

        video, ground_truth = (
            video[random_start_t : random_start_t + actual_num_frames],
            ground_truth[random_start_t : random_start_t + actual_num_frames],
        )

        yield video, ground_truth


seq_iterator = get_seq_iterator()
ct_scan, ground_truth = next(seq_iterator)
print(ground_truth)

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
im, anim = plot_animation(ct_scan, fig, ax)
plt.show()


N_DISCRETE_ACTIONS = 2
HEIGHT = 128
WIDTH = 128
N_CHANNELS = 1


class EnvBinaryClassification(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, seq_iterator):
        super(EnvBinaryClassification, self).__init__()
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8
        )

        self.seq_iterator = seq_iterator
        self.reset()

    def reset(self):
        # Get the next image sequence and ground_truth.
        # Every time we call reset() we use a different image sequence, determined by the next item in self.seq_iterator.
        seq, ground_truth = self.seq_iterator.next()
        self.seq = seq
        self.num_frames = seq.shape[0]
        self.ground_truth = ground_truth
        self.current_frame = 0
        self.predictions = []

        observation = get_observation(self.seq, self.current_frame)
        return observation

    def actions(self):
        return ["diastole", "systole"]

    def step(self, action):
        self.predictions.append(action)
        observation = get_observation(self.seq, self.current_frame + 1)
        reward = get_reward(
            self.current_frame, action, self.predictions, self.ground_truth
        )
        done = self.current_frame == self.num_frames

        # Go to the next frame
        self.current_frame += 1
        return observation, reward, done, None

    def render(self, mode="human"):
        pass

    def close(self):
        pass
