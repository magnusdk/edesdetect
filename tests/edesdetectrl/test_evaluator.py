import edesdetectrl.core.evaluator as evaluator


def test_avg_metrics():
    assert {"a": 2.5, "b": 20, "c": 3.5} == evaluator.avg_metrics(
        [
            {"a": 1, "b": 15, "c": 3},
            {"a": 4, "b": 25, "c": 4},
        ]
    )


def test_evaluator_steps():
    class MockedEvaluation:
        num_evaluations = 0

        def mock_evaluate(self, params, config):
            assert params == "PARAMS"
            assert config == "CONFIG"
            self.num_evaluations += 1
            return {}

    class MockedVariableSource:
        def get_variables(self):
            return "PARAMS"

    delta_episodes = 3
    e = evaluator.Evaluator(
        MockedVariableSource(),
        "CONFIG",
        delta_episodes=delta_episodes,
        metrics_logger=lambda _metrics, _episode: None,
        use_multiprocessing=False,
    )

    mocked_evaluation = MockedEvaluation()
    evaluator.evaluate = mocked_evaluation.mock_evaluate

    for episode in range(1, delta_episodes + 1):
        print(episode, mocked_evaluation.num_evaluations)
        assert 0 == mocked_evaluation.num_evaluations
        e.step(episode, episode)

    for episode in range(delta_episodes + 1, 2 * delta_episodes + 1):
        assert 1 == mocked_evaluation.num_evaluations
        e.step(episode, episode)

    assert 2 == mocked_evaluation.num_evaluations
