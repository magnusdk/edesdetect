# Integration test for checkpoints. It runs very slow, but hey — at least it's automated.
# 
# We test that parameters and train monitor counters are properly saved and restored.
#
# As of now it is tested by actually training an RL agent on the FrozenLakeNonSlippery-v0 
# gym environment. There may be better ways of testing this, but I wanted it to work in 
# situations that are as close to real life situations as possible. Also, I'm new to coax.

import shutil

import gym
import coax
import optax
import haiku as hk
import jax
import jax.numpy as jnp
import jax.tree_util
import chex
import pytest

import edesdetectrl.util.checkpoints as cp


def setup_rl_stuff():
    def func_q(S, A, is_training):
        seq = hk.Sequential(
            (
                hk.Linear(8),
                jax.nn.relu,
                hk.Linear(1, w_init=jnp.zeros),
                jnp.ravel,
            )
        )
        X = jnp.concatenate((S, A), axis=-1)
        return seq(X)

    # Set up RL stuff
    env = coax.wrappers.TrainMonitor(gym.make("FrozenLakeNonSlippery-v0"))
    q = coax.Q(func_q, env)
    pi = coax.EpsilonGreedy(q, epsilon=0.1)
    q_targ = q.copy()
    qlearning = coax.td_learning.QLearning(
        q, q_targ=q_targ, optimizer=optax.adam(0.001)
    )
    tracer = coax.reward_tracing.NStep(n=1, gamma=0.9)
    buffer = coax.experience_replay.SimpleReplayBuffer(capacity=100)
    return env, q, pi, q_targ, qlearning, tracer, buffer


def train_for_n_iterations(n, env, q, pi, q_targ, qlearning, tracer, buffer):
    for _ in range(n):
        s = env.reset()

        for t in range(env.spec.max_episode_steps):
            a = pi(s)
            s_next, r, done, _ = env.step(a)

            # add transition to buffer
            tracer.add(s, a, r, done)
            while tracer:
                transition = tracer.pop()
                buffer.add(transition)

            # update
            if len(buffer) >= 4:
                transition_batch = buffer.sample(batch_size=4)
                metrics = qlearning.update(transition_batch)
                env.record_metrics(metrics)

            # periodically sync target model
            if env.ep % 10 == 0:
                q_targ.soft_update(q, tau=0.5)

            if done:
                break

            s = s_next

    env.close()


def counters_equal(counters1, counters2):
    for attr in [
        "T",
        "ep",
        "t",
        "G",
        "avg_G",
        "_n_avg_G",
        "_ep_starttime",
        # "_ep_actions", # This one is an object and is therefore less trivially compared.
        "_tensorboard_dir",
        "_period",
    ]:
        if counters1[attr] != counters2[attr]:
            return False
    return True


def test_checkpoint_manager_integration():
    checkpoint_dir = "tests/out/"
    checkpoint_path = checkpoint_dir + "checkpoint"

    # Set up checkpoint manager
    env, q, pi, q_targ, qlearning, tracer, buffer = setup_rl_stuff()
    cp_mgr = cp.CheckpointManager(q, env, checkpoint_path)

    # Test that params and counters doesn't change when a checkpoint has not been created yet.
    orig_model_params = q.params
    orig_train_monitor_counters = env.get_counters()
    cp_mgr.restore_latest()  # No checkpoints have been created yet at this point.

    chex.assert_trees_all_close(orig_model_params, q.params)
    assert counters_equal(orig_train_monitor_counters, env.get_counters())

    train_for_n_iterations(100, env, q, pi, q_targ, qlearning, tracer, buffer)

    model_params_after_training = q.params
    train_monitor_counters_after_training = env.get_counters()
    # Assert that the parameters really change. This is not really what we are testing for here, but is a nice sanity check.
    # TODO: This is a flaky assertion and will sometimes fail...
    with pytest.raises(AssertionError):
        chex.assert_trees_all_close(model_params_after_training, orig_model_params)
    assert not counters_equal(
        train_monitor_counters_after_training, orig_train_monitor_counters
    )

    # Create a checkpoint
    cp_mgr.save_checkpoint()

    # Overwrite algorithm stuff. Basically the same as stopping and restarting the script.
    env, q, pi, q_targ, qlearning, tracer, buffer = setup_rl_stuff()
    cp_mgr = cp.CheckpointManager(q, env, checkpoint_path)

    # Another sanity check asserting that the model and counters have been reset.
    with pytest.raises(AssertionError):
        chex.assert_trees_all_close(q.params, model_params_after_training)
    assert not counters_equal(env.get_counters(), train_monitor_counters_after_training)

    cp_mgr.restore_latest()

    # After restoring the checkpoint the parameters should be the same as when we stopped training.
    model_params_after_restored_checkpoint = q.params
    train_monitor_counters_after_restored_checkpoint = env.get_counters()
    chex.assert_trees_all_close(
        model_params_after_restored_checkpoint, model_params_after_training
    )
    assert counters_equal(
        train_monitor_counters_after_restored_checkpoint,
        train_monitor_counters_after_training,
    )

    # A final sanity check that ensures that the model and counters can still be updated after being restored from a checkpoint.
    train_for_n_iterations(100, env, q, pi, q_targ, qlearning, tracer, buffer)
    with pytest.raises(AssertionError):
        chex.assert_trees_all_close(q.params, model_params_after_restored_checkpoint)
    assert not counters_equal(
        env.get_counters(), train_monitor_counters_after_restored_checkpoint
    )
    assert (
        env.get_counters()["T"] > train_monitor_counters_after_restored_checkpoint["T"]
    )

    # Cleanup
    shutil.rmtree(checkpoint_dir, ignore_errors=True)
