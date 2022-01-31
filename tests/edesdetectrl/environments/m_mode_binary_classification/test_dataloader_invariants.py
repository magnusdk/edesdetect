from edesdetectrl.dataloaders.echonet import Echonet
from edesdetectrl.environments.m_mode_binary_classification import (
    EDESMModeClassification_v0,
)


def test_observation_shape():
    """Test that no performance regressions have been introduced."""
    dataloader = Echonet("TRAIN")
    env = EDESMModeClassification_v0(dataloader, "simple")

    for _ in range(len(dataloader)):
        overview_obs, m_mode_obs = env.reset()
        assert overview_obs.shape == (2, 112, 112)
        assert m_mode_obs.shape == (9, 31, 64)
