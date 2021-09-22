from typing import Iterator, List

import acme
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
from acme.agents.jax.dqn import learning_lib, losses
from acme.jax import networks as networks_lib
from acme.jax import variable_utils
from reverb.platform.checkpointers_lib import DefaultCheckpointer

# TODO: Move configuration out and into a configuration map/object, such as acme.agents.jax.dqn.config.DQNConfig

EPSILON: float = 0.05  # Action selection via epsilon-greedy policy.
SEED: int = 1  # Random seed.

# Learning rule
LEARNING_RATE: float = 1e-3  # Learning rate for Adam optimizer.
DISCOUNT: float = 0.99  # Discount rate applied to value per timestep.
N_STEP: int = 5  # N-step TD learning.
TARGET_UPDATE_PERIOD: int = 100  # Update target network every period.
MAX_GRADIENT_NORM: float = jnp.inf  # For gradient clipping.
MAX_ABS_REWARD: float = 1  # TODO: Check this out!
HUBER_LOSS_PARAMETER: float = 1  # TODO: Check this out!

# Replay options
BATCH_SIZE: int = 256  # Number of transitions per batch.
MIN_REPLAY_SIZE: int = 1_000  # Minimum replay size.
MAX_REPLAY_SIZE: int = 1_000_000  # Maximum replay size.
replay_table_name: str = adders_reverb.DEFAULT_PRIORITY_TABLE
IMPORTANCE_SAMPLING_EXPONENT: float = 0.2  # Importance sampling for replay.
PRIORITY_EXPONENT: float = 0.6  # Priority exponent for replay.
prefetch_size: int = 4  # Prefetch size for reverb replay performance.
SAMPLES_PER_INSERT: float = 0.5  # Ratio of learning samples to insert.
# Rate to be used for the SampleToInsertRatio rate limitter tolerance.
# See a formula in make_replay_tables for more details.
samples_per_insert_tolerance_rate: float = 0.1

# How many gradient updates to perform per learner step.
NUM_SGD_STEPS_PER_STEP: int = 1


def get_reverb_replay(
    environment_spec: specs.EnvironmentSpec,
    checkpoints_dir: str,
    extra_spec: types.NestedSpec = (),
    prefetch_size: int = 4,
    replay_table_name: str = adders_reverb.DEFAULT_PRIORITY_TABLE,
) -> replay.ReverbReplay:
    """Creates a single-process replay infrastructure from an environment spec."""
    # Parsing priority exponent to determine uniform vs prioritized replay
    if PRIORITY_EXPONENT is None:
        sampler = reverb.selectors.Uniform()
        priority_fns = {replay_table_name: lambda x: 1.0}
    else:
        sampler = reverb.selectors.Prioritized(PRIORITY_EXPONENT)
        priority_fns = None

    # Create a replay server to add data to. This uses no limiter behavior in
    # order to allow the Agent interface to handle it.
    replay_table = reverb.Table(
        name=replay_table_name,
        sampler=sampler,
        remover=reverb.selectors.Fifo(),
        max_size=MAX_REPLAY_SIZE,
        rate_limiter=reverb.rate_limiters.MinSize(MIN_REPLAY_SIZE),
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
        client, N_STEP, DISCOUNT, priority_fns=priority_fns
    )

    # The dataset provides an interface to sample from replay.
    data_iterator = datasets.make_reverb_dataset(
        table=replay_table_name,
        server_address=address,
        batch_size=BATCH_SIZE,
        prefetch_size=prefetch_size,
    ).as_numpy_iterator()
    return replay.ReverbReplay(server, adder, data_iterator, client=client)


def get_learner(
    network: networks_lib.FeedForwardNetwork,
    data_iterator: Iterator[reverb.ReplaySample],
    replay_client: reverb.Client,  # This one is actually optional, but we will be using reverb, so…
    random_key: networks_lib.PRNGKey,
) -> acme.Learner:

    loss_fn = losses.PrioritizedDoubleQLearning(
        discount=DISCOUNT,
        importance_sampling_exponent=IMPORTANCE_SAMPLING_EXPONENT,
        max_abs_reward=MAX_ABS_REWARD,
        huber_loss_parameter=HUBER_LOSS_PARAMETER,
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(MAX_GRADIENT_NORM),
        optax.adam(LEARNING_RATE),
    )

    counter = None  # TODO: Not really sure what this is.
    logger = None  # TODO: Should we do something more fancy with this?

    learner = learning_lib.SGDLearner(
        network,
        loss_fn,
        optimizer,
        data_iterator,
        TARGET_UPDATE_PERIOD,
        random_key,
        replay_client,
        counter,
        logger,
        NUM_SGD_STEPS_PER_STEP,
    )
    return learner


def get_actor(
    network: networks_lib.FeedForwardNetwork,
    random_key: networks_lib.PRNGKey,
    variable_client: variable_utils.VariableClient,
    adder: adders.Adder,
):
    def policy(
        params: networks_lib.Params,
        key: jnp.ndarray,
        observation: jnp.ndarray,
    ) -> jnp.ndarray:
        action_values = network.apply(params, observation)
        return rlax.epsilon_greedy(EPSILON).sample(key, action_values)

    actor = actors.FeedForwardActor(
        policy=policy,
        random_key=random_key,
        variable_client=variable_client,
        adder=adder,
    )
    return actor


class DQN(agent.Agent, core.Saveable):
    def __init__(
        self,
        network: networks_lib.FeedForwardNetwork,
        reverb_replay: replay.ReverbReplay,
    ):
        # Generate RNG keys for the Learner and Actor
        key_learner, key_actor = jax.random.split(jax.random.PRNGKey(SEED))

        # Create Learner
        learner = get_learner(
            network,
            reverb_replay.data_iterator,
            reverb_replay.client,
            key_learner,
        )

        # Create actor
        actor = get_actor(
            network,
            key_actor,
            variable_utils.VariableClient(learner, ""),
            reverb_replay.adder,
        )

        # At least batch size and at least minimum replay size.
        min_observations = max(BATCH_SIZE, MIN_REPLAY_SIZE)
        # How often we run the learner step. At least* batch size because that is how many transitions are sampled at each step.
        # SAMPLES_PER_INSERT decides how many learner steps will be performed per insertion of batch size transitions.
        # Example: SAMPLES_PER_INSERT=0.5 means that two batch-sizes worth of transitions must be submitted before a learner step will be performed.
        observations_per_step = BATCH_SIZE / SAMPLES_PER_INSERT
        super().__init__(actor, learner, min_observations, observations_per_step)

    def get_variables(self, names=[""]) -> List[List[np.ndarray]]:
        # Names arg is unused, so [""] doesn't mean anything
        return super().get_variables(names)[0]

    def save(self):
        return self._learner.save()

    def restore(self, state):
        print("RESTORING STATE:", state.steps)
        return self._learner.restore(state)
