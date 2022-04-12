import pickle
from collections import defaultdict

import acme.core
import edesdetectrl.util.dm_env as util_dm_env
import jax.numpy as jnp
from acme import specs, types
from acme.jax import networks as networks_lib
from acme.jax import types as jax_types
from edesdetectrl.dataloaders.echonet import Echonet
from edesdetectrl.environments import generate_trajectory_using_actor
from edesdetectrl.environments.vanilla_binary_classification import (
    VanillaBinaryClassificationBase_v0,
)
from edesdetectrl.nets import simple_dqn_network, transform
from jax import random
from jax.tree_util import tree_map


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


def evaluate_with_params(params_path, split):
    base_env = VanillaBinaryClassificationBase_v0("simple")
    env = util_dm_env.GymWrapper(base_env)
    env_spec = specs.make_environment_spec(env)
    network = transform(env_spec, simple_dqn_network(env_spec))
    with open(params_path, "rb") as f:
        params, state = pickle.load(f)
    rng_key = random.PRNGKey(1337)
    actor = ActorImpl(network, params, state, rng_key)

    evaluations = {}
    for data_item in Echonet(split).get_generator(cycle=False):
        if data_item.extra_frames_left >= 3 and data_item.extra_frames_right >= 3:
            base_env.video = data_item
            trajectory = generate_trajectory_using_actor(env, actor)
            evaluations[data_item.name] = trajectory.all_metrics()
    return evaluations
