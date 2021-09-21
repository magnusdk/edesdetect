from collections import namedtuple

import jax.numpy as jnp
import sklearn.metrics as metrics

TrajectoryItem = namedtuple("TrajectoryItem", ["s", "a", "r", "q_values", "env_info"])

# TODO: This Trajectory class is very coupled with BinaryClassification + Q-learning.
# We should probably think a bit more about how to decouple parts of it.
class Trajectory(list):
    """A `Trajectory` is just a list of `TrajectoryItem`s.

    This class simply extends the list class with some helper functions for trajectories."""

    def __init__(self, *trajectory_items):
        super().__init__(*trajectory_items)

    def states(self):
        return [item.s for item in self]

    def actions(self):
        return [item.a for item in self]

    def rewards(self):
        return [item.r for item in self]

    def q_values(self):
        return [item.q_values for item in self]

    def env_info(self):
        return [item.env_info for item in self]

    def _labels_and_predictions(self):
        ground_truths = jnp.array(
            [env_info["ground_truth_phase"] for env_info in self.env_info()]
        )
        actions = jnp.array(self.actions())
        return ground_truths, actions

    def accuracy(self):
        ground_truths, actions = self._labels_and_predictions()
        return metrics.accuracy_score(ground_truths, actions)

    def balanced_accuracy(self):
        ground_truths, actions = self._labels_and_predictions()
        return metrics.balanced_accuracy_score(ground_truths, actions, adjusted=True)

    def recall(self):
        ground_truths, actions = self._labels_and_predictions()
        return metrics.recall_score(
            ground_truths, actions, labels=[0, 1], average="micro"
        )

    def precision(self):
        ground_truths, actions = self._labels_and_predictions()
        return metrics.precision_score(
            ground_truths, actions, labels=[0, 1], average="micro"
        )

    def f1(self):
        ground_truths, actions = self._labels_and_predictions()
        return metrics.f1_score(ground_truths, actions, labels=[0, 1], average="micro")


class GenerateTrajectoryMixin:
    def generate_trajectory_using_q(self, q):
        current_state = self.reset()
        done = False

        trajectory = Trajectory()
        while not done:
            qs = q(current_state)
            a = jnp.argmax(qs)
            next_state, r, done, info = self.step(a)

            trajectory.append(TrajectoryItem(current_state, a, r, qs, info))
            current_state = next_state

        return trajectory
