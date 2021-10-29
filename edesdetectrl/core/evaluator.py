import multiprocessing
from concurrent import futures
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Callable

import acme
import edesdetectrl.environments.binary_classification
import edesdetectrl.nets as nets
import edesdetectrl.util.dm_env as util_dm_env
import gym
import mlflow
from edesdetectrl import environments
from edesdetectrl.agents import dqn
from edesdetectrl.core import stepper
from edesdetectrl.dataloaders.echonet import Echonet


def avg_metrics(all_metrics: dict) -> dict:
    result = {}
    for k in all_metrics[0].keys():
        avg = sum(map(lambda m: m[k], all_metrics)) / len(all_metrics)
        result[k] = avg
    return result


def mlflow_metrics_logger(metrics, episode):
    mlflow.log_metrics(metrics, step=episode)


class Evaluator(stepper.Stepper):
    def __init__(
        self,
        variable_source: acme.VariableSource,
        config: dqn.DQNConfig,
        delta_episodes: int,
        metrics_logger: Callable[[dict, int], None] = mlflow_metrics_logger,
        use_multiprocessing: bool = True,
    ):
        self._variable_source = variable_source
        self._config = config
        self._delta_episodes = delta_episodes
        self._metrics_logger = metrics_logger
        self._use_multiprocessing = use_multiprocessing
        if use_multiprocessing:
            self._process_pool_executor = ProcessPoolExecutor(
                max_workers=5,
                # We must use "spawn" mp-context, otherwise the program crashes.
                # Default "fork" mp-context doesn't play well with multi-threaded programs, with JAX is.
                mp_context=multiprocessing.get_context("spawn"),
            )

    def step(self, episode, _):
        if episode % self._delta_episodes == 0:
            if self._use_multiprocessing:
                evaluate_and_log_metrics(
                    self._variable_source.get_variables(),
                    self._config,
                    episode,
                    self._process_pool_executor,
                    self._metrics_logger,
                )
            else:
                metrics = evaluate(self._variable_source.get_variables(), self._config)
                mlflow.log_metrics(metrics, step=episode)

    def shutdown(self):
        self._process_pool_executor.shutdown(wait=True)


def evaluate_and_log_metrics(
    params,
    config: dqn.DQNConfig,
    episode: int,
    executor: futures.Executor,
    metrics_logger: Callable[[dict, int], None],
):
    future = executor.submit(evaluate, params, config)
    log_metrics = lambda future: metrics_logger(future.result(), step=episode)
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
        network = nets.simple_dqn_network(env_spec)
        actor = dqn.get_evaluation_actor(network, params)

        metrics = []
        for _ in range(len(echonet)):
            trajectory = environments.generate_trajectory_using_actor(env, actor)
            metrics.append(trajectory.all_metrics())

        return avg_metrics(metrics)
