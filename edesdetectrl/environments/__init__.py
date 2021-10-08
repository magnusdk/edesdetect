import math
from collections import namedtuple

import acme.core
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
            [env_info["ground_truth"] for env_info in self.env_info()]
        )
        actions = jnp.array(self.actions())
        return ground_truths, actions

    def sum_rewards(self):
        return sum(self.rewards())

    def accuracy(self):
        ground_truths, actions = self._labels_and_predictions()
        return metrics.accuracy_score(ground_truths, actions)

    def balanced_accuracy(self):
        ground_truths, actions = self._labels_and_predictions()
        bas = metrics.balanced_accuracy_score(ground_truths, actions, adjusted=True)
        if math.isinf(bas) or math.isnan(bas):
            return -1
        else:
            return bas

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

    def all_metrics(self):
        return {
            "sum_rewards": self.sum_rewards(),
            "accuracy": self.accuracy(),
            "balanced_accuracy": self.balanced_accuracy(),
            "recall": self.recall(),
            "precision": self.precision(),
            "f1": self.f1(),
        }


def generate_trajectory_using_q(env, q) -> Trajectory:
    current_state = env.reset()
    done = False

    trajectory = Trajectory()
    while not done:
        qs = q(current_state)
        a = jnp.argmax(qs)
        next_state, r, done, info = env.step(a)

        trajectory.append(TrajectoryItem(current_state, a, r, qs, info))
        current_state = next_state

    return trajectory


def generate_trajectory_using_actor(env, actor: acme.core.Actor) -> Trajectory:
    timestep = env.reset()
    actor.observe_first(timestep)
    trajectory = Trajectory()
    while not timestep.last():
        action = actor.select_action(timestep.observation)
        next_timestep = env.step(action)
        actor.observe(action, next_timestep=next_timestep)
        actor.update()

        trajectory.append(
            TrajectoryItem(
                timestep.observation,
                action,
                next_timestep.reward,
                None,
                next_timestep.info,
            )
        )
        timestep = next_timestep

    return trajectory
