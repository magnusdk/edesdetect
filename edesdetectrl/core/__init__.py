from concurrent.futures.thread import ThreadPoolExecutor

import edesdetectrl.environments.vanilla_binary_classification
import edesdetectrl.nets as nets
import edesdetectrl.util.dm_env as util_dm_env
import gym
from edesdetectrl.agents import dqn
from edesdetectrl.dataloaders.echonet import Echonet


def get_env(config: dqn.DQNConfig, rng_key):
    thread_pool_executor = ThreadPoolExecutor(max_workers=5)
    env = gym.make(
        "VanillaBinaryClassification-v0",
        dataloader=Echonet("TRAIN"),
        get_reward=config.reward_spec,
        rng_key=rng_key,
    )

    def shutdown_env():
        thread_pool_executor.shutdown()

    return util_dm_env.GymWrapper(env), shutdown_env


def get_agent(dqn_config: dqn.DQNConfig, reverb_replay, env_spec):
    network = nets.simple_dqn_network(env_spec)
    agent = dqn.DQN(network, reverb_replay, dqn_config)
    return agent
