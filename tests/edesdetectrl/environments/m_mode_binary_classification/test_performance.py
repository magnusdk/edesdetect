import random as random_std
from timeit import timeit

from edesdetectrl.dataloaders.echonet import Echonet
from edesdetectrl.environments.m_mode_binary_classification import (
    EDESMModeClassification_v0,
)


def random_action(env: EDESMModeClassification_v0):
    env.step(random_std.randint(0, 7))


def test_step_performance():
    """Test that no performance regressions have been introduced."""
    random_std.seed(42)
    env = EDESMModeClassification_v0(Echonet("TRAIN"), "simple")
    env.reset()
    average_load_time_ms = (
        timeit(lambda: env.step(random_std.randint(0, 7)), number=100) * 10
    )
    assert average_load_time_ms <= 12
