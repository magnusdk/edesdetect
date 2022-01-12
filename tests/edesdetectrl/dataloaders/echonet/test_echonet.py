import random as std_random

from edesdetectrl.dataloaders.echonet import Echonet
from jax import random


def test_pixel_normalization():
    seed = std_random.randint(0, 9999999)
    rng_key = random.PRNGKey(seed)
    data_iterator = Echonet("TRAIN").get_random_generator(
        rng_key=rng_key, as_task=False
    )
    for i, data_item in enumerate(data_iterator):
        assert data_item.video.max() == 1
        assert data_item.video.min() == 0
        if i == 100:
            break
