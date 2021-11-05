import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import tensorflow as tf
from jax.lib import xla_bridge as xb

# A bug(?) causes the program to crash because it can not find the tpu_driver when running on the UiO servers.
# It is believed that it caused by some weird caching of XLA (Jax low-level code) backends.
# Calling xb.backends() is a workaround.
xb.backends()
# Another bug(?) causes both Jaxlib and TensorFlow to pre-allocate the memory on a GPU. Let's stop one of them
# (TensorFlow) from doing this.
tf.config.experimental.set_visible_devices([], "GPU")
import logging
import pickle

import acme
import jax
import mlflow
import optax

import edesdetectrl.core as core
from edesdetectrl.agents import dqn
from edesdetectrl.config import config
from edesdetectrl.core import stepper, tracking, train_loop
from edesdetectrl.core.checkpointer import CheckPointer
from edesdetectrl.core.evaluator import Evaluator


def save_variables(variables):
    dir = config["data"]["trained_params_path"]
    os.makedirs(dir, exist_ok=True)
    path = dir + "variables.pkl.lz4"
    with open(path, "wb") as f:
        pickle.dump(variables, f)

    mlflow.log_artifact(path)


def main(dqn_config: dqn.DQNConfig, experiment="default", run_name=None):
    # Initialize random keys
    initial_key = jax.random.PRNGKey(dqn_config.seed)
    (env_rng_key,) = jax.random.split(initial_key, num=1)

    # Set up environment
    env, shutdown_env = core.get_env(dqn_config, env_rng_key)
    env_spec = acme.make_environment_spec(env)

    # Set up agent
    reverb_replay = dqn.get_reverb_replay(
        env_spec,
        config["checkpoints"]["checkpoints_dir"] + "/reverb",
        dqn_config,
    )
    agent = core.get_agent(dqn_config, reverb_replay, env_spec)

    # Set up training-loop related stuff
    evaluator = Evaluator(
        agent,
        dqn_config,
        delta_episodes=20000,
        min_steps=dqn_config.min_replay_size,
    )
    # Checkpointing
    mlflow_initializer = tracking.MLflowInitializer(
        experiment, run_name, dqn_config.as_dict()
    )
    # checkpointer = CheckPointer(
    #    agent,
    #    reverb_replay,
    #    mlflow_initializer,
    #    config["checkpoints"]["checkpoints_dir"],
    #    30,
    # )
    run_id = mlflow_initializer.start_run()
    # checkpointer.set_run_id(run_id)
    episode_stepper = evaluator  # stepper.Combined(checkpointer, evaluator)

    # Train the agent
    train_loop.train_loop(
        env,
        agent,
        dqn_config.num_episodes,
        # start_episode=checkpointer._last_checkpointed_episode(),
        episode_stepper=episode_stepper,
    )

    # Save variables and shut down
    save_variables(agent.get_variables())
    del reverb_replay, agent
    evaluator.shutdown()
    # checkpointer.shutdown()
    shutdown_env()
    mlflow.end_run()

    return run_id


if __name__ == "__main__":
    # Set logging level so that we can see training logs
    logging.basicConfig(level=logging.INFO, force=True)

    dqn_config = dqn.DQNConfig(
        seed=4239814,
        discount=0,
        reward_spec="simple",
        # min_replay_size=1,
        batch_size=512,
        num_episodes=8000,
        learning_rate=1e-3
        # [
        #    optax.piecewise_constant_schedule,
        #    1e-1,
        #    {
        #        50: 1e-2,
        #        100: 1e-3,
        #        200: 1e-4,
        #        300: 1e-5,
        #    },
        # ],
    )
    main(dqn_config)
