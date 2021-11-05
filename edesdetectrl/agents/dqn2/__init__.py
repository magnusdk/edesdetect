from typing import Callable

import dm_env
import gym
import haiku as hk
import jax
import jax.numpy as jnp
from acme import specs, wrappers
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.jax.layouts import distributed_layout
from acme.utils import loggers
from edesdetectrl.agents.dqn2 import config


def get_environment(is_eval) -> dm_env.Environment:
    # Load the gym environment.
    environment = gym.make("MountainCar-v0")

    # Make sure the environment obeys the dm_env.Environment interface.
    environment = wrappers.GymWrapper(environment)
    # Clip the action returned by the agent to the environment spec.
    environment = wrappers.CanonicalSpecWrapper(environment, clip=True)
    environment = wrappers.SinglePrecisionWrapper(environment)


def get_network(
    environment_spec: specs.EnvironmentSpec,
) -> networks_lib.FeedForwardNetwork:
    def forward_fn(obs):
        f = hk.Sequential(
            [
                hk.Linear(16),
                jax.nn.relu,
                hk.Linear(16),
                jax.nn.relu,
                hk.Linear(environment_spec.actions.num_values),
            ]
        )
        return f(obs)

    forward_fn = hk.without_apply_rng(hk.transform(forward_fn))

    dummy_obs = utils.zeros_like(environment_spec.observations)
    dummy_obs = utils.add_batch_dim(dummy_obs)
    network = networks_lib.FeedForwardNetwork(
        lambda rng: forward_fn.init(rng, dummy_obs),
        forward_fn.apply,
    )

    return network


def get_logger(
    save_reverb_logs=False,
    log_every=10.0,
):
    loggers.make_default_logger(
        "learner",
        save_reverb_logs,
        time_delta=log_every,
        asynchronous=True,
        serialize_fn=utils.fetch_devicearray,
        steps_key="learner_steps",
    )


class DistributedDQN(distributed_layout.DistributedLayout):
    def __init__(
        environment_factory: Callable[[bool], dm_env.Environment],
        network_factory: Callable[
            [specs.EnvironmentSpec], networks_lib.FeedForwardNetwork
        ],
        config: config.DQNConfig,
    ):
        seed = 42
        num_actors = 4

        dqn_builder = None

        super().__init__(
            seed=seed,
            environment_factory=lambda: environment_factory(False),
            network_factory=network_factory,
            builder=dqn_builder,
            policy_network=ppo_networks.make_inference_fn,
            evaluator_factories=[
                distributed_layout.default_evaluator(
                    environment_factory=lambda: environment_factory(True),
                    network_factory=network_factory,
                    builder=dqn_builder,
                    policy_factory=eval_policy_factory,
                    log_to_bigtable=save_reverb_logs,
                )
            ],
            num_actors=num_actors,
            prefetch_size=config.prefetch_size,
            max_number_of_steps=max_number_of_steps,
            log_to_bigtable=save_reverb_logs,
            actor_logger_fn=distributed_layout.get_default_logger_fn(
                save_reverb_logs, log_every
            ),
        )
