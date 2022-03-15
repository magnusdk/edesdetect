import pickle
from multiprocessing import Pool
from tqdm.contrib.concurrent import process_map
import acme.core
import edesdetectrl.util.dm_env as util_dm_env
import jax
import jax.numpy as jnp
from acme import specs, types
from acme.jax import networks as networks_lib
from acme.jax import types as jax_types
from edesdetectrl import dataloaders
from edesdetectrl.config import config
from edesdetectrl.dataloaders.echonet import Echonet
from edesdetectrl.environments import generate_trajectory_using_actor
from edesdetectrl.environments.m_mode_binary_classification import (
    EDESMModeClassificationBase_v0,
)
from edesdetectrl.nets import overview_and_m_mode_nets, transform
from jax import random
from jax.tree_util import tree_map
from tqdm import tqdm


def add_batch_dimension(pytree):
    return tree_map(lambda leaf: jnp.expand_dims(leaf, 0), pytree)


class ActorImpl(acme.core.Actor):
    def __init__(
        self,
        network: networks_lib.FeedForwardNetwork,
        params: types.NestedArray,
        state: types.NestedArray,
        initial_rng_key: jax_types.PRNGKey,
    ):
        self.network = network
        self.params = params
        self.state = state
        self.rng_key = initial_rng_key

    def select_action(self, observation):
        self.rng_key, sub_key = random.split(self.rng_key)
        observation = add_batch_dimension(observation)
        result, _ = self.network.apply(self.params, self.state, sub_key, observation)
        return jnp.argmax(result)

    def observe_first(self, timestep):
        pass

    def observe(self, action, next_timestep):
        pass

    def update(self, wait):
        pass


def evaluate(dataitem):
    try:
        base_env = EDESMModeClassificationBase_v0(get_reward="proximity")
        base_env.video = dataitem
        env = util_dm_env.GymWrapper(base_env)
        env_spec = specs.make_environment_spec(env)
        network = transform(env_spec, overview_and_m_mode_nets(env_spec))
        with open("params_116616", "rb") as f:
            params, state = pickle.load(f)
        rng_key = random.PRNGKey(1337)
        actor = ActorImpl(network, params, state, rng_key)

        trajectory = generate_trajectory_using_actor(env, actor)
        return (
            dataitem.name,
            {
                "accuracy": trajectory.accuracy(),
                "balanced_accuracy": trajectory.balanced_accuracy(),
                "recall": trajectory.recall(),
                "precision": trajectory.precision(),
                "f1": trajectory.f1(),
                "num_steps": len(trajectory),
            },
        )
    except:
        return None, None


# 00:10:00  —  ps=20,  cs=10, pf=20
# 00:03:32  —  ps=20,  cs=1,  pf=20
# 00:02:16  —  ps=50,  cs=1,  pf=100
def pre_process_files(dataloader: dataloaders.DataLoader):
    with Pool(processes=50) as pool:
        return dict(
            tqdm(
                pool.imap(
                    evaluate,
                    dataloader.get_generator(cycle=False, prefetch=100),
                    chunksize=1,
                ),
                total=len(dataloader),
            )
        )


def get_pre_processed_scores():
    try:
        scores_path = config["apps"]["performance_insight"]["pre_processed_scores_path"]
        with open(scores_path, "r") as f:
            return eval(f.read().replace("nan", "None"))
    except IOError:
        return {}  # Just an empty dict


if __name__ == "__main__":
    # We only bother to pre-process the TEST part of the dataset.
    echonet = Echonet("TEST")
    scores = pre_process_files(echonet)
    path = config["apps"]["performance_insight"]["pre_processed_scores_path"]
    with open(path, "w") as f:
        f.write(str(scores))
