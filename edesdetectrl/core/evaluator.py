import abc
import multiprocessing
from concurrent import futures
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any

import acme
import edesdetectrl.environments.binary_classification
import edesdetectrl.nets as nets
import edesdetectrl.util.dm_env as util_dm_env
import edesdetectrl.util.mlflow as util_mlflow
import gym
import mlflow
from edesdetectrl import environments
from edesdetectrl.agents import dqn
from edesdetectrl.core import stepper
from edesdetectrl.dataloaders.echonet import Echonet
from mlflow.exceptions import MlflowException


def avg_metrics(all_metrics: dict) -> dict:
    result = {}
    for k in all_metrics[0].keys():
        avg = sum(map(lambda m: m[k], all_metrics)) / len(all_metrics)
        result[k] = avg
    return result


class Logger(abc.ABC):
    @abc.abstractmethod
    def log_metrics(self, metrics, episode):
        pass

    @abc.abstractmethod
    def log_artifact(self, artifact):
        pass

    @abc.abstractmethod
    def get_best_score(self, metric_name):
        pass


class MLFlowLogger(Logger):
    def log_metrics(self, metrics, episode):
        mlflow.log_metrics(metrics, step=episode)

    def log_artifact(self, artifact, filename="variables"):
        util_mlflow.log_artifact(artifact, filename)

    def get_best_score(self, metric_name):
        active_run_id = mlflow.active_run().info.run_id
        client = mlflow.tracking.MlflowClient()
        try:
            # client.get_metric_history(...) raises an exception if the emtric has not been logged before.
            # See issue: https://github.com/mlflow/mlflow/issues/4973
            metric_history = client.get_metric_history(active_run_id, metric_name)
        except MlflowException:
            metric_history = []

        return (
            max(map(lambda x: x.value, metric_history))
            if metric_history
            # If metric_history is an empty list return a really low number.
            else float("-inf")
        )


def log_metrics(
    metrics: dict,
    params: Any,
    episode: int,
    logger: Logger,
):
    # Get current best score
    important_metric = "balanced_accuracy"
    best_score = logger.get_best_score(important_metric)

    # Log metrics
    logger.log_metrics(metrics, episode)

    # Store params if the score is better than the previous best score.
    score = metrics[important_metric]
    if not best_score or score > best_score:
        logger.log_artifact(params)
        logger.log_metrics({"best_" + important_metric: score}, episode)


_ml_flow_logger = MLFlowLogger()


class Evaluator(stepper.Stepper):
    def __init__(
        self,
        variable_source: acme.VariableSource,
        config: dqn.DQNConfig,
        delta_episodes: int,
        logger: Logger = _ml_flow_logger,
        use_multiprocessing: bool = True,
    ):
        self._variable_source = variable_source
        self._config = config
        self._delta_episodes = delta_episodes
        self._logger = logger
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
                    self._logger,
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
    logger: Logger,
):
    future = executor.submit(evaluate, params, config)
    future.add_done_callback(
        lambda future: log_metrics(
            future.result(),
            params,
            episode,
            logger,
        )
    )


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
