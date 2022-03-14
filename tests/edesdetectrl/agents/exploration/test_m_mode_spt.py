"""Testing M-mode exploration using Stateful Property Testing (SPT)."""
import os
from typing import List, Literal

import edesdetectrl.agents.dqn.exploration.m_mode as mex
import edesdetectrl.environments.m_mode_binary_classification as mmode_env
import jax.numpy as jnp
import numpy as np
import pytest
from edesdetectrl.environments.m_mode_binary_classification.line import Bounds, Line
from hypothesis import given
from hypothesis.strategies import lists, sampled_from
from jax import random

# Same as in mmode_env.actions
possible_actions = ["UP", "DOWN", "LEFT", "RIGHT", "ROTATE_LEFT", "ROTATE_RIGHT"]
movement_amount = 1
rotation_amount = 0.1


@given(lists(sampled_from(possible_actions)))
def test_foo(action_sequence: List[str]):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    bounds = Bounds.from_shape((10, 10))
    line = Line.center_of_bounds(bounds, 6)
    state = mex.ExplorationState.initial_exploration_state(line, bounds)
    state.target_line = state.current_line
    print(action_sequence)
    # Move the target line somewhere
    for action in action_sequence:
        state.target_line = perform_action(state.target_line, action)
        print(f"{state.rot_dist + state.pos_dist}, ", end="")


    # We should be able to reach the target line in less than or equal to the number of
    # steps used to move it.
    state.is_exploring = True
    for _ in action_sequence:
        action = mmode_env.iactions[mex.explore(state).tolist()]
        state.current_line = perform_action(state.current_line, action)
    state = mex.maybe_stop_exploration_phase(state, movement_amount, rotation_amount)

    assert not state.is_exploring


def perform_action(line: Line, action: str):
    if action == "UP":
        return line.move_vertically(movement_amount)
    elif action == "DOWN":
        return line.move_vertically(-movement_amount)
    elif action == "LEFT":
        return line.move_horizontally(movement_amount)
    elif action == "RIGHT":
        return line.move_horizontally(-movement_amount)
    elif action == "ROTATE_LEFT":
        return line.rotate(-rotation_amount)
    elif action == "ROTATE_RIGHT":
        return line.rotate(rotation_amount)
