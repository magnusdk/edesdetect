from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import edesdetectrl.environments.m_mode_binary_classification as m_mode_env
import jax
import jax.numpy as jnp
import jax.random as random
from edesdetectrl.environments.m_mode_binary_classification.line import (
    Bounds,
    Line,
    MModeLine,
)
from edesdetectrl.util import sharp_jax
from edesdetectrl.util.pytrees import register_pytree_node_dataclass
from jax._src.random import KeyArray


@register_pytree_node_dataclass
class ExplorationState:
    current_line: Line
    bounds: Bounds
    is_exploring: bool
    target_line: Optional[Line] = None
    # Target line length doesn't matter, but is required.
    line_length: int = 10

    @property
    def pos_dist(self):
        return jnp.sqrt(
            (self.target_line.center.x - self.current_line.center.x) ** 2
            + (self.target_line.center.y - self.current_line.center.y) ** 2
        )

    @property
    def rot_dist(self):
        rot_diff = self.target_line.rotation - self.current_line.rotation
        # Add pi before mod and remove it afterwards to transform it into the interval [-pi, pi].
        return (rot_diff + jnp.pi) % (jnp.pi * 2) - jnp.pi

    @staticmethod
    def initial_exploration_state(line: Line, bounds: Bounds) -> "ExplorationState":
        # The line is just a placeholder. It is not used because is_exploring is False,
        # and will be overridden when exploration phase is started.
        return ExplorationState(
            current_line=line,
            bounds=bounds,
            is_exploring=False,
            target_line=Line(0, 0, 1, 1),
        )


def maybe_start_exploration_phase(
    state: ExplorationState, key: KeyArray, exploration_epsilon: float
) -> ExplorationState:
    """Start exploring with probability exploration_epsilon if not already exploring."""
    k1, k2 = random.split(key, 2)
    return sharp_jax.where(
        jnp.logical_and(
            random.uniform(k1) <= exploration_epsilon,
            jnp.logical_not(state.is_exploring),
        ),
        ExplorationState(
            current_line=state.current_line,
            bounds=state.bounds,
            is_exploring=True,
            target_line=Line.random_within_bounds(k2, state.bounds, state.line_length),
        ),
        state,
    )


def maybe_stop_exploration_phase(
    state: ExplorationState, pos_step_size: float, rot_step_size: float
) -> ExplorationState:
    """Stop exploring when target_line is reached (when the distance is less than the step size)."""
    return sharp_jax.where(
        jnp.logical_and(state.pos_dist < pos_step_size, state.rot_dist < rot_step_size),
        ExplorationState.initial_exploration_state(state.current_line, state.bounds),
        state,
    )


def get_movement_action(line: Line, target_line: Line) -> int:
    """Return the movement action that takes the line closer to the target line."""
    diff = jnp.array(line.center) - jnp.array(target_line.center)
    # Dot products (dp), vertical (v) and horizontal (h).
    dp_v = jnp.dot(diff, jnp.array(target_line.direction))
    dp_h = jnp.dot(diff, jnp.array(target_line.perpendicular_direction))
    return jnp.where(
        jnp.abs(dp_v) > jnp.abs(dp_h),
        jnp.where(dp_v < 0, m_mode_env.actions["DOWN"], m_mode_env.actions["UP"]),
        jnp.where(dp_h > 0, m_mode_env.actions["LEFT"], m_mode_env.actions["RIGHT"]),
    )


def get_rotation_action(rot_dist: float) -> int:
    """Return the rotation action that takes the line closer to the target line."""
    return jnp.where(
        rot_dist > 0,
        m_mode_env.actions["ROTATE_LEFT"],
        m_mode_env.actions["ROTATE_RIGHT"],
    )


def explore(state: ExplorationState) -> int:
    """Explore by trying to match the target line in exploration state.

    Either move (up, down, left, or right) or rotate (left or right), depending on if
    the distance between the line-center positions are greater than the difference in
    angles or vice-versa."""
    return jnp.where(
        jnp.abs(state.pos_dist) > jnp.abs(state.rot_dist),
        get_movement_action(state.current_line, state.target_line),
        get_rotation_action(state.rot_dist),
    )


def get_policy(
    base_policy: Callable,
    exploration_epsilon: float,
    pos_step_size: float,
    rot_step_size: float,
):
    def policy(
        key: KeyArray,
        action_values: jnp.ndarray,
        state: ExplorationState,
    ) -> Tuple[int, ExplorationState]:
        """With a probability of exploration_epsilon, select a target position and rotation
        and move towards it until it has been reached. Otherwise, follow base_policy."""
        state = state.maybe_start_exploration_phase(
            key, exploration_epsilon
        ).maybe_stop_exploration_phase(pos_step_size, rot_step_size)
        return (
            jnp.where(
                state.is_exploring,
                explore(state, state.current_line),
                base_policy(key, action_values),
            ),
            state,
        )

    return policy


# JITed in acme.agents.jax.actors.GenericActor
