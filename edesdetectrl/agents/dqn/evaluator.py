import time
from typing import Callable

import dm_env
import edesdetectrl.util.mlflow as util_mlflow
import jax
import numpy as np
from acme import core, specs
from acme.agents.jax import builders
from acme.jax import networks as networks_lib
from acme.jax import types
from acme.jax.layouts.distributed_layout import NetworkFactory, PolicyFactory
from acme.utils import counting
from acme.utils.loggers import base
from edesdetectrl.agents.dqn import loggers
from edesdetectrl.environments import generate_trajectory_using_actor
from edesdetectrl.environments.m_mode_binary_classification import iactions
from edesdetectrl.util import gpu


def evaluate_and_get_metrics(
    actor: core.Actor,
    num_samples: int,
    environment: dm_env.Environment,
):
    # Evaluate actor on all samples
    all_metrics = []
    all_gaafd = []
    all_action_distributions = []
    for _ in range(num_samples):
        trajectory = generate_trajectory_using_actor(environment, actor)
        all_metrics.append(trajectory.balanced_accuracy())
        all_gaafd.append(trajectory.gaafd())
        all_action_distributions.append(trajectory.action_distribution())

    # Return averaged metrics
    action_distribution = {}
    for distr in all_action_distributions:
        for k, v in distr.items():
            action_distribution[k] = v + action_distribution.get(k, 0)
    action_distribution = {k: v / num_samples for k, v in action_distribution.items()}
    return np.mean(all_metrics), np.mean(all_gaafd), action_distribution


class Evaluator(core.Worker):
    def __init__(
        self,
        actor: core.Actor,
        environment: dict,
        logger: base.Logger,
        counter: counting.Counter,
        evaluator_counter_key: str,
        log_params_artifact: bool = False,
    ):
        self.actor = actor
        self.environment = environment
        self.logger = logger
        self.counter = counter
        self.evaluator_counter_key = evaluator_counter_key
        self.log_params_artifact = log_params_artifact

    def run(self):

        print("Waiting a bit so that batchnorm state has time to be updated.")
        print(
            "TODO: This is too hacky. How to ensure that state is not all zeros before starting to evaluate?"
        )
        import time

        time.sleep(0.5 * 60)  # 5 minutes
        self.actor.update(wait=True)
        print("Starting to evaluate now.")

        environment = self.environment["env"]
        split = self.environment["split"]
        num_samples = self.environment["num_samples"]

        while True:
            counts = self.counter.get_counts()
            learner_steps = counts.get("learner_steps", 0)

            # Get the most updated parameters
            self.actor.update()
            # And possibly log them before evaluation
            if self.log_params_artifact:
                util_mlflow.log_artifact(self.actor._params, f"params_{learner_steps}")

            time_before = time.time()
            balanced_accuracy, gaafd, action_distribution = evaluate_and_get_metrics(
                self.actor, num_samples, environment
            )

            # It's a code-smell to depend on m-mode env for getting the names of 
            # actions, but deadline is approaching and this works.
            presented_action_distribution = {
                (split + "_" + iactions[k]): v for k, v in action_distribution.items()
            }

            elapsed_seconds = time.time() - time_before
            result = {
                (split + "_elapsed_seconds"): elapsed_seconds,
                (split + "_episodes_per_second"): (2 * num_samples) / elapsed_seconds,
                (split + "_balanced_accuracy"): balanced_accuracy,
                (split + "_gaafd"): gaafd,
                "learner_steps": learner_steps,
                **presented_action_distribution,
            }

            self.logger.write(result)

            # Synchronize evaluator steps with learner steps
            previous_evaluator_steps = counts.get(self.evaluator_counter_key, 0)
            diff = learner_steps - previous_evaluator_steps
            self.counter.increment(**{self.evaluator_counter_key: diff})


def get_evaluator_factory(
    # environment_factory returns two environments:
    # One for validation and one for training.
    # These have keys "validate" and "train", respectively.
    environment_factory: Callable[[], dict],
    network_factory: NetworkFactory,
    builder: builders.GenericActorLearnerBuilder,
    policy_factory: PolicyFactory,
    logger_fn,
    log_params_artifact: bool = False,
) -> types.EvaluatorFactory:
    """Returns a default evaluator process."""

    def evaluator(
        random_key: networks_lib.PRNGKey,
        variable_source: core.VariableSource,
        counter: counting.Counter,
        evaluator_counter_key: str,
    ):
        """The evaluation process."""
        gpu.disable_tensorflow_gpu_usage()
        environment = environment_factory()
        # Environment spec is the same for validation and training.
        env_spec = specs.make_environment_spec(environment["env"])

        # Create environment and evaluator networks
        networks = network_factory(env_spec)
        actor = builder.make_actor(
            random_key, policy_factory(networks), variable_source=variable_source
        )

        # Create logger and counter.
        logger = logger_fn()

        return Evaluator(
            actor,
            environment,
            logger,
            counter,
            evaluator_counter_key,
            log_params_artifact=log_params_artifact,
        )

    return evaluator
