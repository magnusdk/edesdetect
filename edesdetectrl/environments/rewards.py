from typing import Sequence

from edesdetectrl.environments.base import BinaryClassificationBaseEnv

MOVEMENT_REWARD = 0.0  # float(-0.1)


def proximity_reward_impl(prediction: int, frame: int, ground_truth: Sequence[int]):
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
        return -float(min(closest_left, closest_right))
    elif closest_left is not None:
        return -float(closest_left)
    elif closest_right is not None:
        return -float(closest_right)
    else:  # There are no ground truth for the prediction in this sequence — give big penalty.
        return float(-len(ground_truth))


def proximity_reward(env: BinaryClassificationBaseEnv, prediction: int) -> float:
    """Return 1.0 if the prediction was correct, else a number negatively
    proportional to the distance to the closest ground truth that matches
    the prediction.

    That is, if the agent predicted Diastole, but the closest Diastole
    frame is 4 frames before the current frame then the reward is -3.
    If it was 5 frames before then the reward would be -4. Distance of 6
    gives -5, etc."""
    if prediction not in (0, 1):
        return MOVEMENT_REWARD
    ground_truth_frame = env.current_frame - env.video.ground_truth_start
    return proximity_reward_impl(prediction, ground_truth_frame, env.video.ground_truth)


def simple_reward(env: BinaryClassificationBaseEnv, prediction: int) -> float:
    """Return 1.0 if the prediction was correct, else -1.0."""
    ground_truth_frame = env.current_frame - env.video.ground_truth_start
    if prediction not in (0, 1):
        return MOVEMENT_REWARD
    return 1.0 if prediction == env.video.ground_truth[ground_truth_frame] else -1.0

