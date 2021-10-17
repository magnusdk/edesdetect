import dataclasses
from typing import Iterator, List

import acme
import edesdetectrl.acme_agents.extensions as extensions
import jax
import jax.numpy as jnp
import numpy as np
import optax
import reverb
import rlax
from acme import adders, core, datasets, specs, types
from acme.adders import reverb as adders_reverb
from acme.agents import agent, replay
from acme.agents.jax import actors
from acme.agents.jax.dqn import config, learning_lib, losses
from acme.jax import networks as networks_lib
from acme.jax import variable_utils
from reverb.platform.checkpointers_lib import DefaultCheckpointer


@dataclasses.dataclass
class DQNConfig:

    # Original license:
    # Copyright 2018 DeepMind Technologies Limited. All rights reserved.
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #     http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.

    """Configuration options for DQN agent."""
    epsilon: float = 0.05  # Action selection via epsilon-greedy policy.
    # TODO(b/191706065): update all clients and remove this field.
    seed: int = 1  # Random seed.

    # Learning rule
    learning_rate: float = 1e-4  # Learning rate for Adam optimizer.
    discount: float = 0.99  # Discount rate applied to value per timestep.
    n_step: int = 1  # N-step TD learning.
    target_update_period: int = 100  # Update target network every period.
    max_gradient_norm: float = jnp.inf  # For gradient clipping.
    max_abs_reward: float = 10  # TODO: Check this out!
    huber_loss_parameter: float = 1  # TODO: Check this out!

    # Replay options
    batch_size: int = 256  # Number of transitions per batch.
    min_replay_size: int = 1_000  # Minimum replay size.
    max_replay_size: int = 500_000  # Maximum replay size.
    replay_table_name: str = adders_reverb.DEFAULT_PRIORITY_TABLE
    importance_sampling_exponent: float = 0.2  # Importance sampling for replay.
    priority_exponent: float = 0.6  # Priority exponent for replay.
    prefetch_size: int = 4  # Prefetch size for reverb replay performance.
    samples_per_insert: float = 0.5  # Ratio of learning samples to insert.
    # Rate to be used for the SampleToInsertRatio rate limitter tolerance.
    # See a formula in make_replay_tables for more details.
    samples_per_insert_tolerance_rate: float = 0.1

    # How many gradient updates to perform per learner step.
    num_sgd_steps_per_step: int = 1

    # Training loop
    # Not much improvement has been observed after 2000 episodes.
    num_episodes: int = 5000

    # Environment
    reward_spec: str = "distance"

    def as_dict(self):
        return dataclasses.asdict(self)


class FixedParams(variable_utils.VariableClient):
    """Implementation of VariableClient but with unchanging parameters."""

    def __init__(self, params):
        self._params = params

    def update(self, _):
        pass

    def update_and_wait(self):
        pass

    @property
    def params(self):
        return self._params


def get_reverb_replay(
    environment_spec: specs.EnvironmentSpec,
    checkpoints_dir: str,
    config: DQNConfig,
    extra_spec: types.NestedSpec = (),
    prefetch_size: int = 4,
    replay_table_name: str = adders_reverb.DEFAULT_PRIORITY_TABLE,
) -> replay.ReverbReplay:
    """Creates a single-process replay infrastructure from an environment spec."""
    # Parsing priority exponent to determine uniform vs prioritized replay
    if config.priority_exponent is None:
        sampler = reverb.selectors.Uniform()
        priority_fns = {replay_table_name: lambda x: 1.0}
    else:
        sampler = reverb.selectors.Prioritized(config.priority_exponent)
        priority_fns = None

    # Create a replay server to add data to. This uses no limiter behavior in
    # order to allow the Agent interface to handle it.
    replay_table = reverb.Table(
        name=replay_table_name,
        sampler=sampler,
        remover=reverb.selectors.Fifo(),
        max_size=config.max_replay_size,
        rate_limiter=reverb.rate_limiters.MinSize(config.min_replay_size),
        signature=adders_reverb.NStepTransitionAdder.signature(
            environment_spec, extra_spec
        ),
    )

    checkpointer = DefaultCheckpointer(checkpoints_dir)
    server = reverb.Server(
        [replay_table],
        port=None,
        checkpointer=checkpointer,
    )

    # The adder is used to insert observations into replay.
    address = f"localhost:{server.port}"
    client = reverb.Client(address)
    adder = adders_reverb.NStepTransitionAdder(
        client, config.n_step, config.discount, priority_fns=priority_fns
    )

    # The dataset provides an interface to sample from replay.
    data_iterator = datasets.make_reverb_dataset(
        table=replay_table_name,
        server_address=address,
        batch_size=config.batch_size,
        prefetch_size=prefetch_size,
    ).as_numpy_iterator()
    return replay.ReverbReplay(server, adder, data_iterator, client=client)


def get_learner(
    network: networks_lib.FeedForwardNetwork,
    data_iterator: Iterator[reverb.ReplaySample],
    replay_client: reverb.Client,  # This one is actually optional, but we will be using reverb, so…
    random_key: networks_lib.PRNGKey,
    config: DQNConfig,
) -> acme.Learner:

    loss_fn = losses.PrioritizedDoubleQLearning(
        discount=config.discount,
        importance_sampling_exponent=config.importance_sampling_exponent,
        max_abs_reward=config.max_abs_reward,
        huber_loss_parameter=config.huber_loss_parameter,
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_gradient_norm),
        optax.adam(config.learning_rate),
    )

    counter = None  # TODO: Not really sure what this is.
    logger = None  # TODO: Should we do something more fancy with this?

    learner = learning_lib.SGDLearner(
        network,
        loss_fn,
        optimizer,
        data_iterator,
        config.target_update_period,
        random_key,
        replay_client,
        counter,
        logger,
        config.num_sgd_steps_per_step,
    )
    return learner


def get_actor(
    network: networks_lib.FeedForwardNetwork,
    random_key: networks_lib.PRNGKey,
    variable_client: variable_utils.VariableClient,
    adder: adders.Adder,
    epsilon: float,
) -> core.Actor:
    def policy(
        params: networks_lib.Params,
        key: jnp.ndarray,
        observation: jnp.ndarray,
    ) -> jnp.ndarray:
        action_values = network.apply(params, observation)
        return rlax.epsilon_greedy(epsilon).sample(key, action_values)

    actor = actors.FeedForwardActor(
        policy=policy,
        random_key=random_key,
        variable_client=variable_client,
        adder=adder,
    )
    return actor


class DQN(agent.Agent, core.Saveable, extensions.Evaluatable):
    def __init__(
        self,
        network: networks_lib.FeedForwardNetwork,
        reverb_replay: replay.ReverbReplay,
        config: DQNConfig,
    ):
        self._network = network
        self._reverb_replay = reverb_replay

        # Generate RNG keys for the Learner and Actor
        key_learner, key_actor, key_eval_actor = jax.random.split(
            jax.random.PRNGKey(config.seed), num=3
        )
        # We need to reference key_eval_actor later when get_evaluation_actor is called.
        self._key_eval_actor = key_eval_actor

        # Create Learner
        learner = get_learner(
            network,
            reverb_replay.data_iterator,
            reverb_replay.client,
            key_learner,
            config,
        )

        # Create actor
        actor = get_actor(
            network,
            key_actor,
            variable_utils.VariableClient(learner, ""),
            reverb_replay.adder,
            config.epsilon,
        )

        # At least batch size and at least minimum replay size.
        min_observations = max(config.batch_size, config.min_replay_size)
        # How often we run the learner step. At least* batch size because that is how many transitions are sampled at each step.
        # SAMPLES_PER_INSERT decides how many learner steps will be performed per insertion of batch size transitions.
        # Example: SAMPLES_PER_INSERT=0.5 means that two batch-sizes worth of transitions must be submitted before a learner step will be performed.
        observations_per_step = config.batch_size / config.samples_per_insert
        super().__init__(actor, learner, min_observations, observations_per_step)

    def get_variables(self, names=[""]) -> List[List[np.ndarray]]:
        # Names arg is unused, so [""] doesn't mean anything
        return super().get_variables(names)[0]

    def save(self):
        return self._learner.save()

    def restore(self, state):
        return self._learner.restore(state)

    def get_evaluation_actor(self):
        """Return an actor that uses a greedy policy."""

        def policy(
            params: networks_lib.Params,
            key: jnp.ndarray,
            observation: jnp.ndarray,
        ) -> jnp.ndarray:
            action_values = self._network.apply(params, observation)
            return rlax.greedy().sample(key, action_values)

        actor = actors.FeedForwardActor(
            policy=policy,
            random_key=self._key_eval_actor,
            variable_client=FixedParams(self.get_variables()),
            adder=self._reverb_replay.adder,
        )

        (self._key_eval_actor,) = jax.random.split(self._key_eval_actor, num=1)
        return actor
