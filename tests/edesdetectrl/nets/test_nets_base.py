import edesdetectrl.environments.m_mode_binary_classification
import edesdetectrl.environments.vanilla_binary_classification
import edesdetectrl.util.dm_env as util_dm_env
import gym
from acme import specs
from edesdetectrl.dataloaders.echonet import Echonet
from edesdetectrl.nets import (
    mobilenet,
    overview_and_m_mode_nets,
    simple_dqn_network,
    transform,
)


def test_creating_networks():
    """Test that instantiating and transforming networks doesn't crash for any environment."""
    env_name = "VanillaBinaryClassification-v0"
    env = util_dm_env.GymWrapper(
        gym.make(env_name, dataloader=Echonet("VAL"), get_reward="simple")
    )
    env_spec = specs.make_environment_spec(env)
    transform(env_spec, simple_dqn_network(env_spec))
    transform(env_spec, mobilenet(env_spec))

    env_name = "EDESMModeClassification-v0"
    env = util_dm_env.GymWrapper(
        gym.make(env_name, dataloader=Echonet("VAL"), get_reward="simple")
    )
    env_spec = specs.make_environment_spec(env)
    transform(env_spec, overview_and_m_mode_nets(env_spec))
