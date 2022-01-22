import time
from typing import Callable

import dm_env
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
from edesdetectrl.util import gpu


def evaluate_and_get_metrics(
    actor: core.Actor,
    num_samples: int,
    environment: dm_env.Environment,
):
    # Get the most updated parameters
    actor.update()

    # Evaluate actor on all samples
    all_metrics = []
    for _ in range(num_samples):
        trajectory = generate_trajectory_using_actor(environment, actor)
        all_metrics.append(trajectory.balanced_accuracy())

    # Return averaged metrics
    return np.mean(all_metrics)


class Evaluator(core.Worker):
    def __init__(
        self,
        actor: core.Actor,
        environment: dict,
        logger: base.Logger,
    ):
        self.actor = actor
        self.environment = environment
        self.logger = logger

    def run(self):

        print("Waiting a bit so that batchnorm state has time to be updated.")
        print(
            "TODO: This is too hacky. How to ensure that state is not all zeros before starting to evaluate?"
        )
        import time

        time.sleep(120)
        self.actor.update(wait=True)
        print("Starting to evaluate now.")

        environment = self.environment["env"]
        split = self.environment["split"]
        num_samples = self.environment["num_samples"]

        while True:
            time_before = time.time()
            balanced_accuracy = evaluate_and_get_metrics(
                self.actor,
                num_samples,
                environment,
            )
            elapsed_seconds = time.time() - time_before
            result = {
                (split + "_elapsed_seconds"): elapsed_seconds,
                (split + "_episodes_per_second"): (2 * num_samples) / elapsed_seconds,
                (split + "_balanced_accuracy"): balanced_accuracy,
            }

            self.logger.write(result)


def get_evaluator_factory(
    # environment_factory returns two environments:
    # One for validation and one for training.
    # These have keys "validate" and "train", respectively.
    environment_factory: Callable[[], dict],
    network_factory: NetworkFactory,
    builder: builders.GenericActorLearnerBuilder,
    policy_factory: PolicyFactory,
    log_to_bigtable: bool = False,
    logger_fn=None,
) -> types.EvaluatorFactory:
    """Returns a default evaluator process."""

    def evaluator(
        random_key: networks_lib.PRNGKey,
        variable_source: core.VariableSource,
        counter: counting.Counter,
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
        if logger_fn:
            logger = logger_fn()
        else:
            logger = loggers.make_default_logger("evaluator", log_to_bigtable)

        return Evaluator(actor, environment, logger)

    return evaluator
