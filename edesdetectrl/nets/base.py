from typing import Callable
import haiku as hk
import jax
import jax.numpy as jnp
from acme import specs
from acme.jax import networks as networks_lib


def simple_dqn_network(
    env_spec: specs.EnvironmentSpec,
) -> networks_lib.FeedForwardNetwork:
    def feed_forward(S, is_training=None):
        f = hk.Sequential(
            [
                # TODO: Fix pre-processing and re-add it here.
                # pre_process_frames,
                hk.Conv2D(16, kernel_shape=8, stride=4),
                jax.nn.relu,
                hk.Conv2D(32, kernel_shape=4, stride=2),
                jax.nn.relu,
                hk.Flatten(),
                hk.Linear(256),
                jax.nn.relu,
                hk.Linear(env_spec.actions.num_values, w_init=jnp.zeros),
            ]
        )
        return f(S)  # Output shape: (batch_size, num_actions=2)

    return feed_forward


def mobilenet(
    env_spec: specs.EnvironmentSpec,
) -> networks_lib.FeedForwardNetwork:
    def feed_forward(S, is_training=True):
        f = hk.nets.MobileNetV1(num_classes=env_spec.actions.num_values)
        return f(S, is_training)

    return feed_forward


def generate_dummy_observation(env_spec_observations):
    if isinstance(env_spec_observations, tuple):
        dummy_obs = [o.generate_value() for o in env_spec_observations]
        batched_dummy_obs = [jnp.expand_dims(o, 0) for o in dummy_obs]
        return tuple(batched_dummy_obs)
    else:
        dummy_obs = env_spec_observations.generate_value()
        batched_dummy_obs = jnp.expand_dims(dummy_obs, 0)
        return batched_dummy_obs


def transform(
    env_spec: specs.EnvironmentSpec, f: Callable
) -> networks_lib.FeedForwardNetwork:
    network_hk = hk.transform_with_state(f)
    dummy_obs = generate_dummy_observation(env_spec.observations)
    network = networks_lib.FeedForwardNetwork(
        init=lambda rng: network_hk.init(rng, dummy_obs),
        apply=network_hk.apply,
    )
    return network
