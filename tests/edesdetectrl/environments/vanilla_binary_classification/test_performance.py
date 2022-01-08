import timeit

import numpy as np
from edesdetectrl.dataloaders.echonet import Echonet
from edesdetectrl.environments.vanilla_binary_classification import (
    VanillaBinaryClassification_v0,
)


def test_performance_regression():
    # TODO: Issue #39 (https://github.com/magnusdk/edesdetect/issues/39)
    return True
    env = VanillaBinaryClassification_v0(Echonet("TRAIN"), "simple")

    def thunk():
        env.reset()
        done = False
        while not done:
            _, _, done, _ = env.step(0)

    result = timeit.repeat(thunk, number=1, repeat=100)
    print(f"Min: {min(result)*1000:.1f} ms")
    print(f"Max: {max(result)*1000:.1f} ms")
    print(f"Mean: {np.mean(result)*1000:.1f} ms")
    #       No changes    / New async dataloader / Optimized get_observation() / New async dataloader and optimized get_observation()
    # Min:  37.7 ms       / 20.8 ms              / 35.3 ms                     / 0.0 ms
    # Max:  546.8 ms      / 199.8 ms             / 569.3 ms                    / 668.5 ms
    # Mean: 127.1ms       / 79.1 ms              / 112.8 ms                    / 21.3 ms

