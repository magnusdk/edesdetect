import edesdetectrl.agents.dqn.exploration.m_mode as mex
import jax.numpy as jnp
import pytest
from edesdetectrl.environments.m_mode_binary_classification.line import Bounds, Line
from jax import random
import os
import numpy as np

# @pytest.fixture
def state():
    # Run the tests on CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    bounds = Bounds.from_shape((112, 112))
    line = Line.center_of_bounds(bounds, 60)
    state = mex.ExplorationState.initial_exploration_state(line, bounds)
    return state


def test_start_exploration_phase(state: mex.ExplorationState):
    assert not state.is_exploring
    key = random.PRNGKey(42)
    state = mex.maybe_start_exploration_phase(state, key, 0.0)
    assert not state.is_exploring
    state = mex.maybe_start_exploration_phase(state, key, 1.0)
    assert state.is_exploring


def test_stop_exploration_phase(state: mex.ExplorationState):
    state.is_exploring = True
    state.target_line = state.current_line.move_horizontally(0.02)
    state = mex.maybe_stop_exploration_phase(state, 0.01, 0.01)
    assert state.is_exploring

    state.is_exploring = True
    state.target_line = state.current_line.move_horizontally(0.01)
    state = mex.maybe_stop_exploration_phase(state, 0.01, 0.01)
    assert not state.is_exploring

    state.is_exploring = True
    state.target_line = state.current_line.rotate(0.02)
    state = mex.maybe_stop_exploration_phase(state, 0.01, 0.01)
    assert state.is_exploring

    state.is_exploring = True
    state.target_line = state.current_line.rotate(0.009)
    state = mex.maybe_stop_exploration_phase(state, 0.01, 0.01)
    assert not state.is_exploring


def test_rot_dist(state: mex.ExplorationState):
    state.target_line = state.current_line
    assert state.rot_dist == 0.0, "Rotation distance is 0 when the lines are equal"

    state.target_line = state.current_line.rotate(1.0)
    assert np.allclose(
        state.rot_dist, 1.0
    ), "Rotation distance is equal to the amount rotated by, when the original target was equal to the current line"

    state.target_line = state.current_line.rotate(-1.0)
    assert np.allclose(
        state.rot_dist, -1.0
    ), "Rotation distance is equal to the amount rotated by, when the original target was equal to the current line"

    state.target_line = state.current_line.rotate(3 / 2 * np.pi)
    assert np.allclose(
        state.rot_dist, -np.pi / 2
    ), "Going beyond positive pi in difference means that it is approaching from the negative direction"

    state.target_line = state.current_line.rotate(-3 / 2 * np.pi)
    assert np.allclose(
        state.rot_dist, np.pi / 2
    ), "Going beyond negative pi in difference means that it is approaching from the positive direction"

    state.target_line = state.current_line.rotate(2 * np.pi)
    assert np.allclose(
        state.rot_dist, 0.0, atol=1.0e-6
    ), "Rotating by 2 pi means we're back where we started"
    state.target_line = state.current_line.rotate(-2 * np.pi)
    assert np.allclose(
        state.rot_dist, 0.0, atol=1.0e-6
    ), "Rotating by -2 pi means we're back where we started"


test_rot_dist(state())
# test_start_exploration_phase(state())
# test_stop_exploration_phase(state())
    