from concurrent.futures.thread import ThreadPoolExecutor

import gym
import numpy as np
from gym import spaces

N_DISCRETE_ACTIONS = 2
HEIGHT = 112
WIDTH = 112
N_PREV_AND_NEXT_FRAMES = 3
N_CHANNELS = 2 * N_PREV_AND_NEXT_FRAMES + 1


def get_reward(current_frame, current_prediction, all_predictions, ground_truths):
    # Is the current prediction correct?
    r1 = 1 if current_prediction == ground_truths[current_frame] else 0

    return r1


def get_observation(seq, frame):
    assert frame >= N_PREV_AND_NEXT_FRAMES
    assert frame <= seq.shape[0] - N_PREV_AND_NEXT_FRAMES

    return seq[frame - N_PREV_AND_NEXT_FRAMES : frame + N_PREV_AND_NEXT_FRAMES + 1]


class EDESClassificationBase_v0(gym.Env):
    """Base class for ED/ES Binary Classification environment.

    Constructor takes a single video and ground truth list.
    For an environment that uses random videos instead, see EDESClassificationRandomVideos_v0"""

    metadata = {"render.modes": ["human"]}

    def __init__(self, seq=None, ground_truth=None):
        super(EDESClassificationBase_v0, self).__init__()
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(HEIGHT, WIDTH, N_CHANNELS), dtype=np.float32
        )
        self._set_seq_and_ground_truth(seq, ground_truth)
        self.reset()

    def _set_seq_and_ground_truth(self, seq, ground_truth):
        assert (
            seq is not None and seq.shape[0] >= N_CHANNELS
        ), f"Video must have more than {N_CHANNELS} frames."
        assert (
            ground_truth is not None and len(ground_truth) == seq.shape[0]
        ), "Ground truth must have the same number of labels as there are frames in the seq."

        self.seq = seq
        self.ground_truth = ground_truth
        self.num_frames = self.seq.shape[0]

    def reset(self):
        self.current_frame = N_PREV_AND_NEXT_FRAMES
        self.predictions = []
        observation = get_observation(self.seq, self.current_frame)
        return observation

    def step(self, action):
        if not (action == 0 or action == 1):  # 0 -> diastole, 1 -> Systole
            raise ValueError(f"Invalid action: {action}.")

        self.predictions.append(action)
        observation = get_observation(self.seq, self.current_frame + 1)
        reward = get_reward(
            self.current_frame, action, self.predictions, self.ground_truth
        )
        done = self.current_frame == self.num_frames - N_PREV_AND_NEXT_FRAMES - 1
        info = {"ground_truth_phase": self.ground_truth[self.current_frame]}

        # Go to the next frame
        self.current_frame += 1
        return observation, reward, done, info

    def render(self, mode="human"):
        pass

    def close(self):
        pass


class EDESClassificationRandomVideos_v0(EDESClassificationBase_v0):
    def __init__(self, seq_iterator):
        def has_enough_frames(seq_and_labels):
            seq, labels = seq_and_labels
            return seq.shape[0] >= N_CHANNELS

        # Let's filter out all seqs that have less frames than the number of channels.
        self.seq_iterator = filter(has_enough_frames, seq_iterator)
        seq, ground_truth = next(self.seq_iterator)
        super(EDESClassificationRandomVideos_v0, self).__init__(seq, ground_truth)

    def reset(self):
        # Get the next image sequence and ground_truth.
        # Every time we call reset() we use a different image sequence, determined by the next item in self.seq_iterator.
        seq, ground_truth = next(self.seq_iterator)
        parent = super(EDESClassificationRandomVideos_v0, self)
        parent._set_seq_and_ground_truth(seq, ground_truth)
        return parent.reset()


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
