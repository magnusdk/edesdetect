# python3
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

"""Program definition for a distributed layout based on a builder."""

import logging
import time
from typing import Any, Callable, List, Optional, Sequence

import dm_env
import jax
import launchpad as lp
import reverb

from acme import core, environment_loop, specs
from acme.agents.jax import builders
from acme.jax import networks as networks_lib
from acme.jax import savers, types, utils
from acme.utils import counting, loggers, signals

ActorId = int
AgentNetwork = Any
NetworkFactory = Callable[[specs.EnvironmentSpec], AgentNetwork]
PolicyFactory = Callable[[AgentNetwork], Any]
EnvironmentFactory = Callable[[], dm_env.Environment]


def get_default_logger_fn(
    log_to_bigtable: bool = False, log_every: float = 10
) -> Callable[[ActorId], loggers.Logger]:
    """Creates an actor logger."""

    def create_logger(actor_id: ActorId):
        return loggers.make_default_logger(
            "actor",
            save_data=(log_to_bigtable and actor_id == 0),
            time_delta=log_every,
            steps_key="actor_steps",
        )

    return create_logger


class StepsLimiterAndWaiterOfEvaluators:
    """Process that terminates an experiment when `max_steps` is reached and when all
    the evaluator have finished their final evaluation."""

    def __init__(
        self,
        counter: counting.Counter,
        max_steps: int,
        evaluator_counter_keys: List[str],
        steps_key: str = "learner_steps",
    ):
        self._counter = counter
        self._max_steps = max_steps
        self.evaluator_counter_keys = evaluator_counter_keys
        self._steps_key = steps_key

    def run(self):
        """Run steps limiter to terminate an experiment when max_steps is reached."""

        logging.info(
            "StepsLimiter: Starting with max_steps = %d (%s)",
            self._max_steps,
            self._steps_key,
        )
        with signals.runtime_terminator():
            while True:
                # Update the counts.
                counts = self._counter.get_counts()
                learner_num_steps = counts.get(self._steps_key, 0)
                evaluators_num_steps = [
                    counts.get(evaluator_counter_key, 0)
                    for evaluator_counter_key in self.evaluator_counter_keys
                ]

                logging.info(
                    f"StepsLimiter: Reached {learner_num_steps} recorded steps (Evaluators: {evaluators_num_steps})"
                )

                # Check if learner has reached max_steps and all evaluators have
                # reached max_steps.
                all_over_max_steps = learner_num_steps > self._max_steps
                for evaluator_num_steps in evaluators_num_steps:
                    all_over_max_steps = all_over_max_steps and (
                        evaluator_num_steps > self._max_steps
                    )

                if all_over_max_steps:
                    logging.info(
                        f"StepsLimiter: Max steps of {self._max_steps} was reached (learner steps: {learner_num_steps}, evaluator steps: {evaluators_num_steps}), terminating"
                    )
                    # Avoid importing Launchpad until it is actually used.
                    import launchpad as lp  # pylint: disable=g-import-not-at-top

                    lp.stop()

                # Don't spam the counter.
                for _ in range(10):
                    # Do not sleep for a long period of time to avoid LaunchPad program
                    # termination hangs (time.sleep is not interruptible).
                    time.sleep(1)


class DistributedLayout:
    """Program definition for a distributed agent based on a builder."""

    def __init__(
        self,
        seed: int,
        environment_factory: EnvironmentFactory,
        network_factory: NetworkFactory,
        builder: builders.GenericActorLearnerBuilder,
        policy_network: PolicyFactory,
        num_actors: int,
        actor_logger_fn: Callable[[ActorId], loggers.Logger],
        environment_spec: Optional[specs.EnvironmentSpec] = None,
        evaluator_factories: Sequence[types.EvaluatorFactory] = (),
        device_prefetch: bool = True,
        prefetch_size: int = 1,
        log_to_bigtable: bool = False,
        max_number_of_steps: Optional[int] = None,
        workdir: str = "~/acme",
        multithreading_colocate_learner_and_reverb: bool = False,
    ):
        if prefetch_size < 0:
            raise ValueError(f"Prefetch size={prefetch_size} should be non negative")

        self._seed = seed
        self._builder = builder
        self._environment_factory = environment_factory
        self._network_factory = network_factory
        self._policy_network = policy_network
        self._environment_spec = environment_spec
        self._num_actors = num_actors
        self._device_prefetch = device_prefetch
        self._log_to_bigtable = log_to_bigtable
        self._prefetch_size = prefetch_size
        self._max_number_of_steps = max_number_of_steps
        self._workdir = workdir
        self._actor_logger_fn = actor_logger_fn
        self._evaluator_factories = evaluator_factories
        self._multithreading_colocate_learner_and_reverb = (
            multithreading_colocate_learner_and_reverb
        )

    def replay(self):
        """The replay storage."""
        environment_spec = self._environment_spec or specs.make_environment_spec(
            self._environment_factory()
        )
        return self._builder.make_replay_tables(environment_spec)

    def counter(self):
        kwargs = {"directory": self._workdir, "add_uid": self._workdir == "~/acme"}
        return savers.CheckpointingRunner(
            counting.Counter(),
            key="counter",
            subdirectory="counter",
            time_delta_minutes=5,
            **kwargs,
        )

    def learner(
        self,
        random_key: networks_lib.PRNGKey,
        replay: reverb.Client,
        counter: counting.Counter,
    ):
        """The Learning part of the agent."""

        iterator = self._builder.make_dataset_iterator(replay)

        environment_spec = self._environment_spec or specs.make_environment_spec(
            self._environment_factory()
        )

        # Creates the networks to optimize (online) and target networks.
        networks = self._network_factory(environment_spec)

        if self._prefetch_size > 1:
            # When working with single GPU we should prefetch to device for
            # efficiency. If running on TPU this isn't necessary as the computation
            # and input placement can be done automatically. For multi-gpu currently
            # the best solution is to pre-fetch to host although this may change in
            # the future.
            device = jax.devices()[0] if self._device_prefetch else None
            iterator = utils.prefetch(
                iterator, buffer_size=self._prefetch_size, device=device
            )
        else:
            logging.info("Not prefetching the iterator.")

        counter = counting.Counter(counter, "learner")

        learner = self._builder.make_learner(
            random_key, networks, iterator, replay, counter
        )
        kwargs = {"directory": self._workdir, "add_uid": self._workdir == "~/acme"}
        # Return the learning agent.
        return savers.CheckpointingRunner(
            learner,
            key="learner",
            subdirectory="learner",
            time_delta_minutes=5,
            **kwargs,
        )

    def actor(
        self,
        random_key: networks_lib.PRNGKey,
        replay: reverb.Client,
        variable_source: core.VariableSource,
        counter: counting.Counter,
        actor_id: int,
    ) -> environment_loop.EnvironmentLoop:
        """The actor process."""
        adder = self._builder.make_adder(replay)

        # Create environment and policy core.
        environment = self._environment_factory()

        networks = self._network_factory(specs.make_environment_spec(environment))
        policy_network = self._policy_network(networks)
        actor = self._builder.make_actor(
            random_key, policy_network, adder, variable_source
        )

        # Create logger and counter.
        counter = counting.Counter(counter, "actor")
        # Only actor #0 will write to bigtable in order not to spam it too much.
        logger = self._actor_logger_fn(actor_id)
        # Create the loop to connect environment and agent.
        return environment_loop.EnvironmentLoop(environment, actor, counter, logger)

    def coordinator(
        self,
        counter: counting.Counter,
        evaluator_counter_keys: List[str],
        max_actor_steps: int,
    ):
        return StepsLimiterAndWaiterOfEvaluators(
            counter, max_actor_steps, evaluator_counter_keys=evaluator_counter_keys
        )

    def build(self, name="agent", program: Optional[lp.Program] = None):
        """Build the distributed agent topology."""
        if not program:
            program = lp.Program(name=name)

        key = jax.random.PRNGKey(self._seed)

        replay_node = lp.ReverbNode(self.replay)
        with program.group("replay"):
            if self._multithreading_colocate_learner_and_reverb:
                replay = replay_node.create_handle()
            else:
                replay = program.add_node(replay_node)

        with program.group("counter"):
            counter = program.add_node(lp.CourierNode(self.counter))

        learner_key, key = jax.random.split(key)
        learner_node = lp.CourierNode(self.learner, learner_key, replay, counter)
        with program.group("learner"):
            if self._multithreading_colocate_learner_and_reverb:
                learner = learner_node.create_handle()
                program.add_node(
                    lp.MultiThreadingColocation([learner_node, replay_node])
                )
            else:
                learner = program.add_node(learner_node)

        with program.group("evaluator"):
            evaluator_counter_keys = []
            for i, evaluator in enumerate(self._evaluator_factories):
                evaluator_counter_key = f"evaluator_{i}_steps"
                evaluator_counter_keys.append(evaluator_counter_key)

                evaluator_key, key = jax.random.split(key)
                program.add_node(
                    lp.CourierNode(
                        evaluator,
                        evaluator_key,
                        learner,
                        counter,
                        evaluator_counter_key,
                    )
                )

        with program.group("actor"):
            for actor_id in range(self._num_actors):
                actor_key, key = jax.random.split(key)
                program.add_node(
                    lp.CourierNode(
                        self.actor, actor_key, replay, learner, counter, actor_id
                    )
                )

        with program.group("coordinator"):
            program.add_node(
                lp.CourierNode(
                    self.coordinator,
                    counter,
                    evaluator_counter_keys,
                    self._max_number_of_steps,
                )
            )

        return program
