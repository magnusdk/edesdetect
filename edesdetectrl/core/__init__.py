from concurrent.futures.thread import ThreadPoolExecutor

import edesdetectrl.environments.binary_classification
import edesdetectrl.nets as nets
import edesdetectrl.util.dm_env as util_dm_env
import edesdetectrl.util.generators as generators
import gym
from edesdetectrl.agents import dqn
from edesdetectrl.dataloaders.echonet import Echonet


def get_env(config: dqn.DQNConfig, rng_key):
    thread_pool_executor = ThreadPoolExecutor(max_workers=5)
    echonet = Echonet("TRAIN")
    data_iterator = generators.async_buffered(
        echonet.get_random_generator(rng_key),
        thread_pool_executor,
        5,
    )
    env = gym.make(
        "EDESClassification-v0",
        seq_iterator=data_iterator,
        reward=config.reward_spec,
    )

    def shutdown_env():
        thread_pool_executor.shutdown()

    return util_dm_env.GymWrapper(env), shutdown_env


def get_agent(dqn_config: dqn.DQNConfig, reverb_replay, env_spec):
    network = nets.simple_dqn_network(env_spec)
    agent = dqn.DQN(network, reverb_replay, dqn_config)
    return agent
