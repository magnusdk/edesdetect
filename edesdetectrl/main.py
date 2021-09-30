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
import os
import pickle
from concurrent.futures.thread import ThreadPoolExecutor

import acme
import dm_env
import gym
import haiku as hk
from acme import core
from acme.agents import replay
from acme.jax import networks as networks_lib

import edesdetectrl.dataloaders.echonet as echonet
import edesdetectrl.environments.binary_classification as bc
import edesdetectrl.model as model
import edesdetectrl.util.dm_env as util_dm_env
from edesdetectrl.acme_agents import dqn
from edesdetectrl.config import config
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
        checkpoints_dir: str,
        time_delta_minutes: float,
    ) -> None:
        self._agent = agent
        self._reverb_replay = reverb_replay

        os.makedirs(checkpoints_dir, exist_ok=True)
        self._checkpoints_learner_path = checkpoints_dir + "/learner"

        # Restore learner state from checkpoint
        try:
            with open(self._checkpoints_learner_path, "rb") as f:
                learner_state = pickle.load(f)
                self._agent.restore(learner_state)
        except FileNotFoundError:
            # #EAFP (https://devblogs.microsoft.com/python/idiomatic-python-eafp-versus-lbyl/)
            pass

        self._checkpoint_timer = timer.Timer(time_delta_minutes * 60)

    def step(self):
        if self._checkpoint_timer.check():
            self._reverb_replay.server.in_process_client().checkpoint()
            # Learner state is the only thing we need to save.
            learner_state = self._agent.save()
            with open(self._checkpoints_learner_path, "wb") as f:
                pickle.dump(learner_state, f)
            self._checkpoint_timer.reset()

    def _last_checkpointed_episode(self):
        # TODO: Rethink how to do this. Maybe the checkpointer should hold a reference to a training loop object and restore it?
        client = self._reverb_replay.server.in_process_client()
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
        evaluator.step()
        checkpointer.step()


def get_env(thread_pool_executor, split):
    seq_iterator = echonet.get_generator(
        thread_pool_executor,
        volumetracings_csv_file=config["data"]["volumetracings_path"],
        filelist_csv_file=config["data"]["filelist_path"],
        videos_dir=config["data"]["videos_path"],
        split=split,
        buffer_maxsize=5,
    )
    env = gym.make(
        "EDESClassification-v0",
        seq_iterator=seq_iterator,
        get_reward=bc._get_dist_reward,
    )
    return util_dm_env.GymWrapper(env)


def main():
    thread_pool_executor = ThreadPoolExecutor()
    training_env = get_env(thread_pool_executor, "TRAIN")
    validation_env = get_env(thread_pool_executor, "VAL")
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
    reverb_replay = dqn.get_reverb_replay(env_spec, CHECKPOINTS_DIR_REVERB)
    print()
    print("-----")
    print(reverb_replay.server.in_process_client().server_info())
    print("-----")
    print()

    agent = dqn.DQN(network, reverb_replay)
    checkpointer = CheckPointer(agent, reverb_replay, CHECKPOINTS_DIR, 30)

    evaluator = Evaluator(
        validation_env,
        agent,
        n_trajectories=200,
        delta_episodes=1000,
        start_episode=checkpointer._last_checkpointed_episode(),
    )
    # Run training loop
    train_loop(training_env, evaluator, agent, 10000, checkpointer)

    with open(config["data"]["trained_params_path"], "wb") as f:
        network_params = agent.get_variables()
        pickle.dump(network_params, f)


if __name__ == "__main__":
    main()
