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

"""DQN agent implementation."""

import functools
from typing import Callable, Optional

import dm_env
import reverb
import rlax
from acme import core as acme_core
from acme import environment_loop, specs
from acme.agents.jax import actor_core as actor_core_lib
from acme.agents.jax import normalization
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.jax.layouts import distributed_layout
from acme.utils import counting
from edesdetectrl.agents.dqn import builder
from edesdetectrl.agents.dqn import config as dqn_config
from edesdetectrl.agents.dqn import evaluator, loggers, losses
from edesdetectrl.util import gpu
from jax import random

NetworkFactory = Callable[[specs.EnvironmentSpec], networks_lib.FeedForwardNetwork]


def make_inference_fn(
    network: networks_lib.FeedForwardNetwork,
    config: dqn_config.DQNConfig,
    evaluation: bool = False,
) -> actor_core_lib.FeedForwardPolicy:
    """Returns a function to be used for inference by a DQN actor."""

    def inference(
        params: networks_lib.Params,
        state: networks_lib.Params,
        key: networks_lib.PRNGKey,
        observation: networks_lib.Observation,
    ):
        network_key, sample_key = random.split(key)
        action_values, _ = network.apply(
            params, state, network_key, observation, not evaluation  # Training
        )

        if evaluation:
            return rlax.greedy().sample(sample_key, action_values)
        else:
            return rlax.epsilon_greedy(config.epsilon).sample(sample_key, action_values)

    return inference


class DistributedDQN(distributed_layout.DistributedLayout):
    def __init__(
        self,
        environment_factory: Callable[[bool, Optional[str]], dm_env.Environment],
        network_factory: NetworkFactory,
        config: dqn_config.DQNConfig,
        seed: int,
        num_actors: int,
        tracking_uri,
        experiment,
        run_id,
        normalize_input: bool = False,
        save_reverb_logs: bool = False,
        log_every: float = 10.0,
        max_number_of_steps: Optional[int] = None,
    ):
        learner_logger_fn = functools.partial(
            loggers.make_default_logger,
            "learner",
            tracking_uri,
            experiment,
            run_id,
            save_reverb_logs,
            time_delta=log_every,
            asynchronous=True,
            serialize_fn=utils.fetch_devicearray,
        )
        val_evaluator_logger_fn = functools.partial(
            loggers.make_default_logger,
            "val_evaluator",
            tracking_uri,
            experiment,
            run_id,
            save_reverb_logs,
            time_delta=log_every,
            asynchronous=True,
            serialize_fn=utils.fetch_devicearray,
        )
        train_evaluator_logger_fn = functools.partial(
            loggers.make_default_logger,
            "train_evaluator",
            tracking_uri,
            experiment,
            run_id,
            save_reverb_logs,
            time_delta=log_every,
            asynchronous=True,
            serialize_fn=utils.fetch_devicearray,
        )

        loss_fn = losses.PrioritizedDoubleQLearning(
            discount=config.discount,
            importance_sampling_exponent=config.importance_sampling_exponent,
        )
        dqn_builder = builder.DQNBuilder(config, loss_fn, learner_logger_fn)
        if normalize_input:
            environment_spec = specs.make_environment_spec(environment_factory(False))
            # Two batch dimensions: [num_sequences, num_steps, ...]
            batch_dims = (0, 1)
            dqn_builder = normalization.NormalizationBuilder(
                dqn_builder,
                environment_spec,
                is_sequence_based=True,
                batch_dims=batch_dims,
            )
        eval_policy_factory = lambda network: make_inference_fn(network, config, True)
        super().__init__(
            seed=seed,
            environment_factory=lambda: environment_factory(False),
            network_factory=network_factory,
            builder=dqn_builder,
            policy_network=lambda network: make_inference_fn(network, config, False),
            evaluator_factories=[
                evaluator.get_evaluator_factory(
                    environment_factory=lambda: environment_factory(True, "VAL"),
                    network_factory=network_factory,
                    builder=dqn_builder,
                    policy_factory=eval_policy_factory,
                    log_to_bigtable=save_reverb_logs,
                    logger_fn=val_evaluator_logger_fn,
                ),
                evaluator.get_evaluator_factory(
                    environment_factory=lambda: environment_factory(True, "TRAIN"),
                    network_factory=network_factory,
                    builder=dqn_builder,
                    policy_factory=eval_policy_factory,
                    log_to_bigtable=save_reverb_logs,
                    logger_fn=train_evaluator_logger_fn,
                ),
            ],
            num_actors=num_actors,
            prefetch_size=config.prefetch_size,
            max_number_of_steps=max_number_of_steps,
            log_to_bigtable=save_reverb_logs,
            actor_logger_fn=distributed_layout.get_default_logger_fn(
                save_reverb_logs, log_every
            ),
        )

    def coordinator(self, counter: counting.Counter, max_actor_steps: int):
        gpu.disable_tensorflow_gpu_usage()
        return super().coordinator(counter, max_actor_steps)

    def counter(self):
        gpu.disable_tensorflow_gpu_usage()
        return super().counter()

    def learner(
        self,
        random_key: networks_lib.PRNGKey,
        replay: reverb.Client,
        counter: counting.Counter,
    ):
        gpu.disable_tensorflow_gpu_usage()
        return super().learner(random_key, replay, counter)

    def actor(
        self,
        random_key: networks_lib.PRNGKey,
        replay: reverb.Client,
        variable_source: acme_core.VariableSource,
        counter: counting.Counter,
        actor_id: int,
    ) -> environment_loop.EnvironmentLoop:
        gpu.disable_tensorflow_gpu_usage()
        return super().actor(random_key, replay, variable_source, counter, actor_id)
