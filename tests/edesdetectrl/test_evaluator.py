from tests.mocks import agents, environments

import edesdetectrl.evaluator as evaluator
import edesdetectrl.util.dm_env as util_dm_env


def test_avg_metrics():
    assert {"a": 2.5, "b": 20, "c": 3.5} == evaluator.avg_metrics(
        [
            {"a": 1, "b": 15, "c": 3},
            {"a": 4, "b": 25, "c": 4},
        ]
    )


na = None  # "Not Applicable". Also, Batman.


def test_evaluator_steps():
    class MockedEvaluation:
        num_evaluations = 0

        def mock_evaluate(self):
            self.num_evaluations += 1

    delta_episodes = 3
    e = evaluator.Evaluator(na, na, na, delta_episodes=delta_episodes)
    mocked_evaluation = MockedEvaluation()
    e.evaluate = mocked_evaluation.mock_evaluate

    for _ in range(delta_episodes):
        assert 0 == mocked_evaluation.num_evaluations
        e.step()

    for _ in range(delta_episodes):
        assert 1 == mocked_evaluation.num_evaluations
        e.step()

    assert 2 == mocked_evaluation.num_evaluations


def test_evaluator_evaluate():
    env = util_dm_env.GymWrapper(environments.SlidingState([0, 1, 0]))
    evaluatable = agents.AlwaysPicksActionEvaluatable(action=0)
    n_trajectories = 100
    e = evaluator.Evaluator(
        env,
        evaluatable,
        n_trajectories,
        delta_episodes=na,
        start_episode=na,
    )

    result = e.evaluate()

    assert result == {
        "sum_rewards": 1.0,
        "accuracy": 0.5,
        "balanced_accuracy": 0.0,
        "recall": 0.5,
        "precision": 0.5,
        "f1": 0.5,
    }
