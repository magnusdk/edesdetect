from collections import namedtuple

import jax.numpy as jnp

TrajectoryItem = namedtuple("TrajectoryItem", ["s", "a", "r", "q_values", "env_info"])


class GenerateTrajectoryMixin:
    def generate_trajectory_using_q(self, q):
        current_state = self.reset()
        done = False

        trajectory = []
        while not done:
            qs = q(current_state)
            a = jnp.argmax(qs)
            next_state, r, done, info = self.step(a)

            trajectory.append(TrajectoryItem(current_state, a, r, qs, info))
            current_state = next_state

        return trajectory
