import tensorflow as tf
from jax.lib import xla_bridge as xb
from mlflow.utils.logging_utils import MLFLOW_LOGGING_STREAM

# A bug(?) causes the program to crash because it can not find the tpu_driver when running on the UiO servers.
# It is believed that it caused by some weird caching of XLA (Jax low-level code) backends.
# Calling xb.backends() is a workaround.
xb.backends()
# Another bug(?) causes both Jaxlib and TensorFlow to pre-allocate the memory on a GPU. Let's stop one of them
# (TensorFlow) from doing this.
tf.config.experimental.set_visible_devices([], "GPU")
import logging
import os
import pickle
from concurrent.futures.thread import ThreadPoolExecutor

import acme
import dm_env
import gym
import haiku as hk
import jax
import mlflow
from acme import core
from acme.agents import replay
from acme.jax import networks as networks_lib

import edesdetectrl.model as model
import edesdetectrl.util.dm_env as util_dm_env
from edesdetectrl import tracking
from edesdetectrl.acme_agents import dqn
from edesdetectrl.config import config
from edesdetectrl.dataloaders.echonet import Echonet
from edesdetectrl.evaluator import Evaluator
from edesdetectrl.util import functional, timer

CHECKPOINTS_DIR = "/scratch/users/magnukva/_checkpoints"
CHECKPOINTS_DIR_REVERB = CHECKPOINTS_DIR + "/reverb"


# TODO: Can this be generalized? Take a second look when other agents are added.
class CheckPointer:
    def __init__(
        self,
        agent: core.Saveable,
        reverb_replay: replay.ReverbReplay,
        mlflow_initializer: tracking.MLflowInitializer,
        checkpoints_dir: str,
        time_delta_minutes: float,
    ) -> None:
        self._agent = agent
        self._reverb_replay = reverb_replay

        os.makedirs(checkpoints_dir, exist_ok=True)
        self._checkpoints_learner_path = checkpoints_dir + "/learner"
        self._checkpoints_misc_path = checkpoints_dir + "/misc"

        # Restore learner state from checkpoint
        try:
            with open(self._checkpoints_learner_path, "rb") as f:
                learner_state = pickle.load(f)
                self._agent.restore(learner_state)
        except FileNotFoundError:
            # #EAFP (https://devblogs.microsoft.com/python/idiomatic-python-eafp-versus-lbyl/)
            pass

        # Restore mlflow state
        try:
            with open(self._checkpoints_misc_path, "rb") as f:
                misc = pickle.load(f)
                mlflow_initializer.set_run_id(misc["mlflow_run_id"])
        except FileNotFoundError:
            pass

        self._checkpoint_timer = timer.Timer(time_delta_minutes * 60)

    def set_run_id(self, run_id):
        """Set the MLflow run_id so that the run can be restored."""
        misc = {}
        # Try to update from disk
        try:
            with open(self._checkpoints_misc_path, "rb") as f:
                misc_from_disk = pickle.load(f)
                misc.update(misc_from_disk)
        except FileNotFoundError:
            pass

        # Set the run_id and write to disk
        misc["mlflow_run_id"] = run_id
        with open(self._checkpoints_misc_path, "wb") as f:
            pickle.dump(misc, f)

    def step(self):
        if self._checkpoint_timer.check():
            # Checkpoint reverb replay
            self._reverb_replay.server.localhost_client().checkpoint()

            # Save learner state
            learner_state = self._agent.save()
            with open(self._checkpoints_learner_path, "wb") as f:
                pickle.dump(learner_state, f)

            self._checkpoint_timer.reset()

    def _last_checkpointed_episode(self):
        # TODO: Rethink how to do this. Maybe the checkpointer should hold a reference to a training loop object and restore it?
        client = self._reverb_replay.server.localhost_client()
        server_info = client.server_info()
        num_episodes = server_info["priority_table"].num_episodes
        return num_episodes


def train_episode(
    env: dm_env.Environment,
    actor: core.Actor,
):
    timestep = env.reset()
    actor.observe_first(timestep)
    while not timestep.last():
        action = actor.select_action(timestep.observation)
        timestep = env.step(action)
        actor.observe(action, next_timestep=timestep)
        actor.update()


def train_loop(
    training_env: dm_env.Environment,
    evaluator: Evaluator,
    actor: core.Actor,
    num_episodes: int,
    checkpointer: CheckPointer,
):
    for episode in range(checkpointer._last_checkpointed_episode(), num_episodes):
        if episode % 100 == 0:
            # TODO: Use logging object.
            logging.info(f"  Episode {episode}/{num_episodes}")
        train_episode(training_env, actor)
        if metrics := evaluator.step():
            mlflow.log_metrics(metrics, step=episode)
        checkpointer.step()


def get_env(reward_spec, split, rng_key, thread_pool_executor):
    echonet = Echonet(split)
    seq_iterator = echonet.get_random_generator(
        rng_key, thread_pool_executor, prefetch=5
    )
    env = gym.make(
        "EDESClassification-v0",
        seq_iterator=seq_iterator,
        reward=reward_spec,
    )
    return util_dm_env.GymWrapper(env)


def main():
    dqn_config = dqn.DQNConfig(discount=0)

    thread_pool_executor = ThreadPoolExecutor()
    training_dataloader_rng_key, validation_dataloader_rng_key = jax.random.split(
        jax.random.PRNGKey(dqn_config.seed), num=2
    )
    training_env = get_env(
        dqn_config.reward_spec, "TRAIN", training_dataloader_rng_key, thread_pool_executor
    )
    validation_env = get_env(
        dqn_config.reward_spec, "VAL", validation_dataloader_rng_key, thread_pool_executor
    )
    env_spec = acme.make_environment_spec(training_env)

    # Create network
    network_hk = functional.chainf(
        model.get_func_approx(env_spec.actions.num_values),
        hk.transform,
        hk.without_apply_rng,
    )
    dummy_obs = env_spec.observations.generate_value()
    network = networks_lib.FeedForwardNetwork(
        init=lambda rng: network_hk.init(rng, dummy_obs),
        apply=network_hk.apply,
    )

    # Set logging level so that we can see training logs
    logging.basicConfig(level=logging.INFO, force=True)

    # Create agent
    reverb_replay = dqn.get_reverb_replay(env_spec, CHECKPOINTS_DIR_REVERB, dqn_config)

    # mlflow tracking set up
    mlflow_initializer = tracking.MLflowInitializer(
        "binary_classification_environment",
        "Distance-based reward 2",
        dqn_config.as_dict(),
    )
    agent = dqn.DQN(network, reverb_replay, dqn_config)
    checkpointer = CheckPointer(
        agent,
        reverb_replay,
        mlflow_initializer,
        CHECKPOINTS_DIR,
        30,
    )

    evaluator = Evaluator(
        validation_env,
        agent,
        n_trajectories=200,
        delta_episodes=100,
        start_episode=checkpointer._last_checkpointed_episode(),
    )
    # Run training loop
    run_id = mlflow_initializer.start_run()
    checkpointer.set_run_id(run_id)
    train_loop(training_env, evaluator, agent, dqn_config.num_episodes, checkpointer)

    with open(config["data"]["trained_params_path"], "wb") as f:
        network_params = agent.get_variables()
        pickle.dump(network_params, f)

    mlflow.log_artifact(config["data"]["trained_params_path"])
    mlflow_initializer.end_run()


if __name__ == "__main__":
    main()
