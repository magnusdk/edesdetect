import haiku as hk
import jax
import jax.numpy as jnp
from acme import specs
from acme.jax import networks as networks_lib


def simple_dqn_network(
    env_spec: specs.EnvironmentSpec,
) -> networks_lib.FeedForwardNetwork:
    def func_approx(S):
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

    network_hk = hk.without_apply_rng(hk.transform(func_approx))
    dummy_obs = env_spec.observations.generate_value()
    network = networks_lib.FeedForwardNetwork(
        init=lambda rng: network_hk.init(rng, dummy_obs),
        apply=network_hk.apply,
    )
    return network
