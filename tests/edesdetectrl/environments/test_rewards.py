import edesdetectrl.environments.rewards as rewards


def test_proximity_reward():
    def gen_rewards(prediction, ground_truth):
        return [
            rewards.proximity_reward_impl(prediction, frame, ground_truth)
            for frame in range(len(ground_truth))
        ]

    ground_truth = [0]
    assert [1] == gen_rewards(
        0, ground_truth
    ), "Get a reward for correctly predicting 0"

    ground_truth = [0, 0]
    assert [1, 1] == gen_rewards(
        0, ground_truth
    ), "Get a reward for correctly predicting 0 for both frames"

    ground_truth = [1]
    assert [-1] == gen_rewards(
        0, ground_truth
    ), "Get a penalty of len(ground_truth) of predicting 0 when there are no 0s in the ground truth"

    ground_truth = [1, 0]
    assert [0, 1] == gen_rewards(
        0, ground_truth
    ), "Get neither reward nor penalty when the wrongly predicted frame is close to a correct one"

    ground_truth = [0, 1, 1, 1, 1]
    assert [1, 0, -1, -2, -3] == gen_rewards(
        0, ground_truth
    ), "Get bigger and bigger penalty the more wrong the prediction is"

    ground_truth = [0, 1, 1, 1, 0]
    assert [1, 0, -1, 0, 1] == gen_rewards(
        0, ground_truth
    ), "A wrong prediction is only as wrong as the distance to the nearest ground truth with the predicted value"

    ground_truth = [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1]
    assert [1, 1, 0, -1, -2, -1, 0, 1, 1, 1, 1, 0, -1] == gen_rewards(0, ground_truth)

    ground_truth = []
    assert [] == gen_rewards(
        0, ground_truth
    ), "Edge case: no ground truths doesn't crash :)"
