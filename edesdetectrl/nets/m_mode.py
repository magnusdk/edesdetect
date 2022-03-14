import haiku as hk
import jax
import jax.numpy as jnp
from acme import specs
from acme.jax import networks as networks_lib


def overview_and_m_mode_nets(
    env_spec: specs.EnvironmentSpec,
) -> networks_lib.FeedForwardNetwork:
    def feed_forward(S, is_training=None):
        overview, m_modes = S[0], S[1]

        overview_f = hk.Sequential(
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
                hk.Linear(env_spec.actions.num_values),
            ]
        )
        # Add batchnorm
        # Remove stride
        # Experiment with mobilenet
        m_modes_f = hk.Sequential(
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
                hk.Linear(env_spec.actions.num_values),
            ]
        )

        combined = jnp.concatenate([overview_f(overview), m_modes_f(m_modes)], axis=-1)
        final_layer = hk.Linear(env_spec.actions.num_values)
        return final_layer(combined)

    return feed_forward
