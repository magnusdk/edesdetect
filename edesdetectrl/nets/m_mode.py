import haiku as hk
import jax
import jax.numpy as jnp
from acme import specs
from acme.jax import networks as networks_lib


def basic_dqn_inspired_cnn():
    return hk.Sequential(
        [
            hk.Conv2D(16, kernel_shape=8, stride=4),
            jax.nn.relu,
            hk.Conv2D(32, kernel_shape=4, stride=2),
            jax.nn.relu,
            hk.Flatten(),
            hk.Linear(256),
            jax.nn.relu,
        ]
    )


def overview_and_m_mode_nets(
    env_spec: specs.EnvironmentSpec,
) -> networks_lib.FeedForwardNetwork:
    def feed_forward(S, is_training=None):
        overview, m_modes, action_history = S[0], S[1], S[2]
        action_history = hk.Flatten()(action_history)

        overview_f = basic_dqn_inspired_cnn()
        m_modes_f = basic_dqn_inspired_cnn()
        action_history_f = hk.nets.MLP([32])
        combined = jnp.concatenate(
            [
                overview_f(overview),
                m_modes_f(m_modes),
                action_history_f(action_history),
            ],
            axis=-1,
        )

        final_layer = hk.nets.MLP([64, env_spec.actions.num_values])
        return final_layer(combined)

    return feed_forward


def overview_and_m_mode_nets_mmode(
    env_spec: specs.EnvironmentSpec,
) -> networks_lib.FeedForwardNetwork:
    def feed_forward(S, is_training=True):
        overview, m_modes, action_history = S[0], S[1], S[2]
        action_history = hk.Flatten()(action_history)

        overview_f = basic_dqn_inspired_cnn()
        m_modes_f = hk.nets.MobileNetV1(num_classes=256)
        action_history_f = hk.nets.MLP([32, 64])
        combined = jnp.concatenate(
            [
                overview_f(overview),
                m_modes_f(m_modes, is_training),
                action_history_f(action_history),
            ],
            axis=-1,
        )

        final_layer = hk.nets.MLP([32, 64, env_spec.actions.num_values])
        return final_layer(combined)

    return feed_forward


def overview_and_m_mode_nets_forked(
    env_spec: specs.EnvironmentSpec,
) -> networks_lib.FeedForwardNetwork:
    def feed_forward(S, is_training=None):
        overview, m_modes, action_history = S[0], S[1], S[2]
        action_history = hk.Flatten()(action_history)

        m_modes_base = hk.Sequential(
            [
                hk.Conv2D(16, 3),
                jax.nn.relu,
                hk.Conv2D(32, 3),
                jax.nn.relu,
            ]
        )(m_modes)

        overview = basic_dqn_inspired_cnn()(overview)
        action_history = hk.nets.MLP([32, 64])(action_history)
        m_modes_movement = hk.Sequential(
            [
                hk.Conv2D(64, 3),
                jax.nn.relu,
                hk.Conv2D(128, 3),
                jax.nn.relu,
                hk.Flatten(),
                hk.Linear(256),
                jax.nn.relu,
            ]
        )(m_modes_base)
        movement = hk.nets.MLP([32, 64, 6])(
            jnp.concatenate(
                [m_modes_movement, overview, action_history],
                axis=-1,
            )
        )

        prediction = hk.Sequential(
            [
                hk.Conv2D(64, kernel_shape=3),
                jax.nn.relu,
                hk.Conv2D(128, kernel_shape=3),
                jax.nn.relu,
                hk.Flatten(),
                hk.Linear(256),
                jax.nn.relu,
                hk.Linear(2),
            ]
        )(m_modes_base)

        return jnp.concatenate([prediction, movement], axis=-1)

    return feed_forward