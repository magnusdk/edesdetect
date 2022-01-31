from timeit import timeit

from edesdetectrl.dataloaders.echonet import Echonet
from jax import random


def test_loading_performance():
    """Test that no performance regressions have been introduced."""
    rng_key = random.PRNGKey(42)
    data_iterator = Echonet("TRAIN").get_random_generator(rng_key)
    average_load_time_ms = timeit(lambda: next(data_iterator), number=1000)

    assert average_load_time_ms <= 20
