from concurrent import futures
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import acme
import gym
import mlflow

import edesdetectrl.environments.binary_classification
import edesdetectrl.model as model
import edesdetectrl.util.dm_env as util_dm_env
from edesdetectrl import environments
from edesdetectrl.acme_agents import dqn
from edesdetectrl.config import config
from edesdetectrl.dataloaders.echonet import Echonet


def avg_metrics(all_metrics: dict) -> dict:
    result = {}
    for k in all_metrics[0].keys():
        avg = sum(map(lambda m: m[k], all_metrics)) / len(all_metrics)
        result[k] = avg
    return result


class Evaluator:
    def __init__(
        self,
        variable_source: acme.VariableSource,
        config: dqn.DQNConfig,
        delta_episodes: int,
        start_episode: int = 0,
    ):
        self._variable_source = variable_source
        self._config = config
        self._delta_episodes = delta_episodes
        self._episode = start_episode
        self._process_pool_executor = ProcessPoolExecutor(
            max_workers=5,
            # We must use "spawn" mp-context, otherwise the program crashes.
            # Default "fork" mp-context doesn't play well with multi-threaded programs, with JAX is.
            mp_context=multiprocessing.get_context("spawn"),
        )

    def step(self):
        self._episode += 1
        if self._episode % self._delta_episodes == 0:
            evaluate_and_log_metrics(
                self._variable_source.get_variables(),
                self._config,
                self._episode,
                self._process_pool_executor,
            )

    def shutdown(self):
        self._process_pool_executor.shutdown(wait=True)


def evaluate_and_log_metrics(
    params, config: dqn.DQNConfig, episode: int, executor: futures.Executor
):
    future = executor.submit(evaluate, params, config)
    log_metrics = lambda future: mlflow.log_metrics(future.result(), step=episode)
    future.add_done_callback(log_metrics)


def evaluate(params, config: dqn.DQNConfig):
    with ThreadPoolExecutor(max_workers=5) as executor:
        echonet = Echonet("VAL")
        env = util_dm_env.GymWrapper(
            gym.make(
                "EDESClassification-v0",
                seq_iterator=echonet.get_generator(executor),
                reward=config.reward_spec,
            )
        )
        env_spec = acme.make_environment_spec(env)
        network = model.as_feed_forward_network(
            model.get_func_approx(env_spec.actions.num_values), env_spec
        )
        actor = dqn.get_evaluation_actor(network, params)

        metrics = []
        for _ in range(len(echonet)):
            trajectory = environments.generate_trajectory_using_actor(env, actor)
            metrics.append(trajectory.all_metrics())

        return avg_metrics(metrics)
