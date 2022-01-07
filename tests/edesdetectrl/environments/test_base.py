import numpy as np
import pytest
from edesdetectrl.environments.base import BinaryClassificationBaseEnv


def test_padding():
    pad_left = 2
    pad_right = 3
    env = BinaryClassificationBaseEnv(
        pad_left,
        pad_right,
        lambda _env: 0,
        lambda _env, _a: 0,
    )

    video_length = 10
    video = np.random.random((video_length, 10, 10))
    ground_truth = [0 for _ in range(video_length)]
    env.video_and_labels = video, ground_truth

    env.reset()
    assert env.current_frame == pad_left, "The first frame is the pad_left-nth one"

    done = False
    while not done:
        _, _, done, _ = env.step(0)
    assert (
        env.current_frame == video_length - pad_right - 1
    ), "There are pad_right remaining frames when the episode is done."


def test_video_and_labels_padding_assertions():
    pad_left = 2
    pad_right = 3
    env = BinaryClassificationBaseEnv(
        pad_left,
        pad_right,
        lambda _env: 0,
        lambda _env, _a: 0,
    )

    # 6 frames are enough since there are enough frames to pad 2 on the left and 3 on the right (2 + current frame + 3 = 6).
    video_length = 6
    video = np.random.random((video_length, 10, 10))
    ground_truth = [0 for _ in range(video_length)]
    env.video_and_labels = video, ground_truth

    # However, 5 frames are not enough.
    video_length = 5
    video = np.random.random((video_length, 10, 10))
    ground_truth = [0 for _ in range(video_length)]
    with pytest.raises(AssertionError):
        env.video_and_labels = video, ground_truth


def test_episode_immediately_done():
    """Assert that episode is immediately done when stepping on a video that has only one frame outside of padding."""
    pad_left = 2
    pad_right = 3
    video_length = pad_left + 1 + pad_right

    def get_observation(env: BinaryClassificationBaseEnv):
        """Return the first frame with padding and the last frame with padding."""
        return (
            env._ground_truth[env.current_frame - pad_left],
            env._ground_truth[env.current_frame + pad_right],
        )

    env = BinaryClassificationBaseEnv(
        pad_left,
        pad_right,
        get_observation,
        lambda _env, _a: 0,
    )

    video = np.random.random((video_length, 10, 10))
    ground_truth = [0 for _ in range(video_length)]
    env.video_and_labels = video, ground_truth
    env.reset()
    _, _, done, _ = env.step(0)
    assert done
