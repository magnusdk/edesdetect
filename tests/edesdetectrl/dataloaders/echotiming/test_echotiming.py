import random

import jax
import numpy as np
from edesdetectrl.dataloaders.echotiming import EchoTiming


def test_keys():
    train_dataset = EchoTiming("TRAIN")
    val_dataset = EchoTiming("VAL")
    test_dataset = EchoTiming("TEST")
    train_avi_names_keys = set(train_dataset.keys)
    val_avi_names_keys = set(val_dataset.keys)
    test_avi_names_keys = set(test_dataset.keys)

    assert all(
        [
            (avi_name not in val_avi_names_keys and avi_name not in test_avi_names_keys)
            for avi_name in train_avi_names_keys
        ]
    ), "TRAIN split videos are not present in VAL or TEST split videos"
    assert all(
        [(avi_name not in val_avi_names_keys) for avi_name in test_avi_names_keys]
    ), "TEST split videos are not present in VAL split videos"

    total_num = (
        len(train_avi_names_keys) + len(val_avi_names_keys) + len(test_avi_names_keys)
    )
    assert np.allclose(
        len(train_avi_names_keys) / total_num, 0.6
    ), "TRAIN split is roughly 60%"
    assert np.allclose(
        len(val_avi_names_keys) / total_num, 0.2
    ), "VAL split is roughly 20%"
    assert np.allclose(
        len(test_avi_names_keys) / total_num, 0.2
    ), "TEST split is roughly 20%"


def test_echotiming_dataloader():
    # Mostly ensure that nothing crashes and burns.
    for item in EchoTiming(None).get_generator(cycle=False):
        assert item.video.ndim == 3
