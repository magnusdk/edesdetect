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

import dataclasses
from typing import Literal

import gym
import jax
import launchpad as lp
import mlflow
from acme import specs
from launchpad.nodes.python.local_multi_processing import PythonProcess

import edesdetectrl.agents.dqn.config as dqn_config
import edesdetectrl.config as general_config
import edesdetectrl.util.dm_env as util_dm_env
from edesdetectrl.agents.dqn import agent
from edesdetectrl.dataloaders.echonet import Echonet
from edesdetectrl.nets import (
    mobilenet,
    overview_and_m_mode_nets,
    simple_dqn_network,
    transform,
)


@dataclasses.dataclass
class ExperimentConfig:
    environment: Literal[
        "VanillaBinaryClassification-v0",
        "EDESMModeClassification-v0",
    ] = "VanillaBinaryClassification-v0"

    network: Literal[
        "simple",
        "mobilenet",
        "m_mode_simple",
    ] = "mobilenet"

    reward_spec: Literal[
        "simple",
        "proximity",
    ] = "simple"


def get_environment_factory(experiment_config: ExperimentConfig, rng_key):
    def environment_factory(is_eval: bool, split="TRAIN"):
        if experiment_config.environment == "VanillaBinaryClassification-v0":
            import edesdetectrl.environments.vanilla_binary_classification
        elif experiment_config.environment == "EDESMModeClassification-v0":
            import edesdetectrl.environments.m_mode_binary_classification

        env = util_dm_env.GymWrapper(
            gym.make(
                experiment_config.environment,
                dataloader=Echonet(split),
                get_reward=experiment_config.reward_spec,
                rng_key=rng_key if split == "TRAIN" else None,
            )
        )

        if is_eval:  # Evaluation expects a dictionary with some additional information.
            return {
                "num_samples": len(Echonet("VAL")),
                "env": env,
                "split": split,
            }
        else:  # Otherwise, just return env.
            return env

    return environment_factory


def get_network_factory(experiment_config: ExperimentConfig):
    def network_factory(env_spec: specs.EnvironmentSpec):
        if experiment_config.network == "simple":
            return transform(env_spec, simple_dqn_network(env_spec))
        elif experiment_config.network == "mobilenet":
            return transform(env_spec, mobilenet(env_spec))
        elif experiment_config.network == "m_mode_simple":
            return transform(env_spec, overview_and_m_mode_nets(env_spec))

    return network_factory


def main():
    mlflow.set_tracking_uri(general_config.config["mlflow"]["tracking_uri"])
    mlflow.set_experiment("dqn_distributed")
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        tracking_id = general_config.config["mlflow"]["tracking_uri"]
        experiment = "dqn_distributed"

        seed = 42
        config = dqn_config.DQNConfig(
            epsilon=0.2,
            learning_rate=1e-3,
            discount=0.97,
            n_step=5,
            num_sgd_steps_per_step=8,
            seed=seed,
        )
        experiment_config = ExperimentConfig(
            environment="EDESMModeClassification-v0",
            network="m_mode_simple",
            reward_spec="proximity",
        )
        mlflow.log_params(dataclasses.asdict(config))
        mlflow.log_params(dataclasses.asdict(experiment_config))

        program = agent.DistributedDQN(
            environment_factory=get_environment_factory(
                experiment_config, jax.random.PRNGKey(seed + 1)
            ),
            network_factory=get_network_factory(experiment_config),
            config=config,
            seed=seed,
            num_actors=6,
            tracking_uri=tracking_id,
            experiment=experiment,
            run_id=run_id,
            max_number_of_steps=4_000_000,
        ).build()

        # Launch experiment.
        no_cuda_devices = PythonProcess(env=dict(CUDA_VISIBLE_DEVICES=""))
        lp.launch(
            program,
            lp.LaunchType.LOCAL_MULTI_PROCESSING,
            local_resources={
                "actor": no_cuda_devices,
                "evaluator": no_cuda_devices,
                "counter": no_cuda_devices,
                "replay": no_cuda_devices,
            },
        )


if __name__ == "__main__":
    main()
