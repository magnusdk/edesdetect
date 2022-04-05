import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import edesdetectrl.environments.m_mode_binary_classification
import edesdetectrl.environments.vanilla_binary_classification
import edesdetectrl.util.dm_env as util_dm_env
import gym
from acme import specs
from edesdetectrl.dataloaders.echonet import Echonet
from edesdetectrl.nets import (
    mobilenet,
    overview_and_m_mode_nets,
    overview_and_m_mode_nets_mmode,
    simple_dqn_network,
    transform,
)
from jax import random


def test_creating_networks():
    """Test that instantiating and transforming networks doesn't crash for any environment."""
    rng_key = random.PRNGKey(42)

    env_name = "VanillaBinaryClassification-v0"
    env = util_dm_env.GymWrapper(
        gym.make(env_name, dataloader=Echonet("VAL"), get_reward="simple")
    )
    env_spec = specs.make_environment_spec(env)
    transform(env_spec, simple_dqn_network(env_spec)).init(rng_key)
    transform(env_spec, mobilenet(env_spec)).init(rng_key)

    env_name = "EDESMModeClassification-v0"
    env = util_dm_env.GymWrapper(
        gym.make(
            env_name, dataloader=Echonet("VAL"), get_reward="simple", rng_key=rng_key
        )
    )
    env_spec = specs.make_environment_spec(env)
    transform(env_spec, overview_and_m_mode_nets(env_spec)).init(rng_key)
    transform(env_spec, overview_and_m_mode_nets_mmode(env_spec)).init(rng_key)
