import haiku as hk
import jax
import jax.numpy as jnp
import coax


def pre_process_frames(frames, dtype="float32"):
    """Pre-process the frames to get position, velocity, acceleration, etc..."""

    # TODO: This is buggy because coax.utils.diff_transform(...) expects the last channel in frames to be the current frame.
    # This is not true for our data, where the current frame is in the middle, and we show the previous and next frames.
    return coax.utils.diff_transform(frames, dtype)


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
