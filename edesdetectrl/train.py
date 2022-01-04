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
from concurrent.futures.thread import ThreadPoolExecutor

import gym
import jax
import launchpad as lp
from acme import specs

import edesdetectrl.agents.dqn.config as dqn_config
import edesdetectrl.util.dm_env as util_dm_env
import edesdetectrl.util.generators as generators
from edesdetectrl.agents.dqn import agent
from edesdetectrl.dataloaders.echonet import Echonet
from edesdetectrl.nets import mobilenet, simple_dqn_network


def get_environment_factory(rng_key):
    def environment_factory(is_eval: bool, split="TRAIN"):
        import edesdetectrl.environments.binary_classification

        thread_pool_executor = ThreadPoolExecutor(max_workers=5)
        if is_eval:
            if split == "VAL":
                data_iterator = generators.async_buffered(
                    Echonet("VAL").get_generator(),
                    thread_pool_executor,
                    5,
                )
                env = gym.make(
                    "EDESClassification-v0",
                    seq_iterator=data_iterator,
                    reward="simple",
                )
            elif split == "TRAIN":
                data_iterator = generators.async_buffered(
                    Echonet("TRAIN").get_random_generator(rng_key),
                    thread_pool_executor,
                    5,
                )
                env = gym.make(
                    "EDESClassification-v0",
                    seq_iterator=data_iterator,
                    reward="simple",
                )

            return {
                "num_samples": len(Echonet("VAL")),
                "env": util_dm_env.GymWrapper(env),
                "split": split,
            }

        else:
            data_iterator = generators.async_buffered(
                Echonet(split).get_random_generator(rng_key),
                thread_pool_executor,
                5,
            )
            env = gym.make(
                "EDESClassification-v0",
                seq_iterator=data_iterator,
                reward="simple",
            )
            return util_dm_env.GymWrapper(env)

    return environment_factory


def network_factory(env_spec: specs.EnvironmentSpec):
    return mobilenet(env_spec)
    #return simple_dqn_network(env_spec)


import mlflow

import edesdetectrl.config as general_config


def main():
    mlflow.set_tracking_uri(general_config.config["mlflow"]["tracking_uri"])
    mlflow.set_experiment("dqn_distributed")
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        tracking_id = general_config.config["mlflow"]["tracking_uri"]
        experiment = "dqn_distributed"

        seed = 42
        config = dqn_config.DQNConfig(
            epsilon=1,
            learning_rate=1e-4,
            discount=0,
            n_step=1,
            num_sgd_steps_per_step=8,
            seed=seed,
        )
        mlflow.log_params(dataclasses.asdict(config))
        environment_factory = get_environment_factory(jax.random.PRNGKey(seed + 1))
        program = agent.DistributedDQN(
            environment_factory=environment_factory,
            network_factory=network_factory,
            config=config,
            seed=seed,
            num_actors=6,
            tracking_uri=tracking_id,
            experiment=experiment,
            run_id=run_id,
            max_number_of_steps=4_000_000,
        ).build()

        # Launch experiment.
        lp.launch(program, lp.LaunchType.LOCAL_MULTI_PROCESSING)


if __name__ == "__main__":
    main()
