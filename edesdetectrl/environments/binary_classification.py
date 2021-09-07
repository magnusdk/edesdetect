from concurrent.futures.thread import ThreadPoolExecutor

import edesdetectrl.environments.mixins as mixins
import gym
import numpy as np
from gym import spaces

N_DISCRETE_ACTIONS = 2
HEIGHT = 112
WIDTH = 112
N_PREV_AND_NEXT_FRAMES = 3
N_CHANNELS = 2 * N_PREV_AND_NEXT_FRAMES + 1


def _get_reward(env, current_prediction):
    # Is the current prediction correct?
    r1 = 1 if current_prediction == env._ground_truth[env.current_frame] else 0

    return r1


def _get_dist_reward_impl(prediction, frame, ground_truth):
    # Find the frame difference between the current frame and the first ground truth that 
    # matches the prediction to the left of the frame.
    closest_left = 0
    while ground_truth[frame - closest_left] != prediction:
        closest_left += 1
        if frame - closest_left < 0:
            closest_left = None
            break

    # Find the frame difference between the current frame and the first ground truth that 
    # matches the prediction to the right of the frame.
    closest_right = 0
    while ground_truth[frame + closest_right] != prediction:
        closest_right += 1
        if frame + closest_right >= len(ground_truth):
            closest_right = None
            break

    # Return the lowest frame difference.
    if closest_left is not None and closest_right is not None:
        return 1 - min(closest_left, closest_right)
    elif closest_left is not None:
        return 1 - closest_left
    elif closest_right is not None:
        return 1 - closest_right
    else:  # There are no ground truth for the prediction in this sequence — give big penalty.
        return -len(ground_truth)


def _get_dist_reward(env, prediction):
    return _get_dist_reward_impl(prediction, env.current_frame, env._ground_truth)


def _get_observation(env):
    frame = env.current_frame
    seq = env._seq
    assert frame >= N_PREV_AND_NEXT_FRAMES
    assert frame <= seq.shape[0] - N_PREV_AND_NEXT_FRAMES

    return seq[frame - N_PREV_AND_NEXT_FRAMES : frame + N_PREV_AND_NEXT_FRAMES + 1]


def _get_mock_observation(env):
    if env._ground_truth[env.current_frame] == 0:
        return np.zeros((HEIGHT, WIDTH, N_CHANNELS))
    else:
        return np.ones((HEIGHT, WIDTH, N_CHANNELS)) * 255


class EDESClassificationBase_v0(gym.Env, mixins.GenerateTrajectoryMixin):
    """Base class for ED/ES Binary Classification environment.

    Constructor takes a single video and ground truth list.
    For an environment that uses random videos instead, see EDESClassificationRandomVideos_v0"""

    metadata = {"render.modes": ["human"]}

    def __init__(self, get_reward=_get_reward, get_observation=_get_observation):
        super(EDESClassificationBase_v0, self).__init__()
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(HEIGHT, WIDTH, N_CHANNELS), dtype=np.float32
        )
        self._seq, self._ground_truth = None, None
        self.get_reward = get_reward
        self.get_observation = get_observation

    def is_ready(self):
        return self._seq is not None and self._ground_truth is not None

    def reset(self):
        assert self.is_ready(), "seq and ground_truth must be set."
        self.current_frame = N_PREV_AND_NEXT_FRAMES
        self.predictions = []
        observation = self.get_observation(self)
        return observation

    def step(self, action):
        if not (action == 0 or action == 1):  # 0 -> diastole, 1 -> Systole
            raise ValueError(f"Invalid action: {action}.")

        self.predictions.append(action)
        reward = self.get_reward(self, action)
        done = self.current_frame == self._seq.shape[0] - N_PREV_AND_NEXT_FRAMES - 1
        info = {"ground_truth_phase": self._ground_truth[self.current_frame]}

        # Go to the next frame
        self.current_frame += 1
        observation = self.get_observation(self)

        return observation, reward, done, info

    def render(self, mode="human"):
        pass

    def close(self):
        pass

    @property
    def seq_and_labels(self):
        return (self._seq, self._ground_truth)

    @seq_and_labels.setter
    def seq_and_labels(self, seq_and_labels):
        seq, ground_truth = seq_and_labels
        assert (
            ground_truth is not None and len(ground_truth) == seq.shape[0]
        ), "Ground truth must have the same number of labels as there are frames in the seq."
        assert (
            seq is not None and seq.shape[0] >= N_CHANNELS
        ), f"Video must have more than {N_CHANNELS} frames."
        self._seq = seq
        self._ground_truth = ground_truth


class EDESClassificationRandomVideos_v0(EDESClassificationBase_v0):
    def __init__(
        self,
        seq_iterator,
        get_reward=_get_reward,
        get_observation=_get_observation,
    ):
        def has_enough_frames(seq_and_labels):
            seq, labels = seq_and_labels
            return seq.shape[0] >= N_CHANNELS

        # Let's filter out all seqs that have less frames than the number of channels.
        self.seq_iterator = filter(has_enough_frames, seq_iterator)
        super(EDESClassificationRandomVideos_v0, self).__init__(
            get_reward, get_observation
        )

    def reset(self):
        # Get the next image sequence and ground_truth.
        # Every time we call reset() we use a different image sequence, determined by the next item in self.seq_iterator.
        self.seq_and_labels = next(self.seq_iterator)
        return super(EDESClassificationRandomVideos_v0, self).reset()


gym.register(
    id="EDESClassification-v0",
    entry_point="edesdetectrl.environments.binary_classification:EDESClassificationRandomVideos_v0",
    max_episode_steps=200,
)


def timeit_test():
    import random
    import timeit

    import edesdetectrl.dataloaders.echonet as echonet
    from edesdetectrl.config import config
    from numpy.core.fromnumeric import mean

    print("Time reseting the environment and stepping through it until done:")
    random.seed(1337)
    volumetracings_csv_file = config["data"]["volumetracings_path"]
    filelist_csv_file = config["data"]["filelist_path"]
    videos_dir = config["data"]["videos_path"]
    split = "TRAIN"
    with ThreadPoolExecutor() as thread_pool_executor:
        seq_iterator = echonet.get_generator(
            thread_pool_executor,
            volumetracings_csv_file,
            filelist_csv_file,
            videos_dir,
            split,
            buffer_maxsize=50,
        )
        env = EDESClassificationRandomVideos_v0(seq_iterator)

        def thunk():
            env.reset()
            done = False
            while not done:
                _, _, done, _ = env.step(0)

        result = timeit.repeat(thunk, number=1, repeat=100)
        print(f"Min: {min(result)*1000:.1f} ms")
        print(f"Max: {max(result)*1000:.1f} ms")
        print(f"Mean: {mean(result)*1000:.1f} ms")
        #       No changes    / New async dataloader / Optimized get_observation() / New async dataloader and optimized get_observation()
        # Min:  37.7 ms       / 20.8 ms              / 35.3 ms                     / 0.0 ms
        # Max:  546.8 ms      / 199.8 ms             / 569.3 ms                    / 668.5 ms
        # Mean: 127.1ms       / 79.1 ms              / 112.8 ms                    / 21.3 ms


# timeit_test()
