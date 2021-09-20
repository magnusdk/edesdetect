import haiku as hk
import jax
import jax.numpy as jnp


def get_func_approx(num_actions):
    def w_init(shape, dtype=None):
        return jnp.zeros(shape, dtype="float32")

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
                hk.Linear(num_actions, w_init=w_init),
            ]
        )
        return f(S)  # Output shape: (batch_size, num_actions=2)

    return func_approx
