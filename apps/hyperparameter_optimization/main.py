import time

import edesdetectrl.util.mlflow as util_mlflow
import nevergrad as ng
import numpy as np
from edesdetectrl import train
from edesdetectrl.agents.dqn import DQNConfig


def evaluate(epsilon, learning_rate, discount, n_step):
    config = DQNConfig(
        epsilon=epsilon,
        learning_rate=learning_rate,
        discount=discount,
        n_step=n_step,
        num_episodes=1000,
        min_replay_size=10000,
    )
    start_time = time.time()
    mlflow_run_id = train.main(config, experiment="nevergrad_hyperparam_opt")
    elapsed_time = time.time() - start_time

    score = util_mlflow.best_score(mlflow_run_id, "balanced_accuracy")
    print(
        f"Elapsed time: {elapsed_time}. Score: {score}.",
        (epsilon, learning_rate, discount, n_step),
    )
    return -score


if __name__ == "__main__":
    np.random.seed(42)

    epsilon_p = ng.p.Scalar(lower=0, upper=1)
    learning_rate_p = ng.p.Log(lower=1e-8, upper=1e-1, exponent=10)
    discount_p = ng.p.Scalar(lower=0, upper=1)
    n_step_p = ng.p.TransitionChoice(range(1, 10))

    instru = ng.p.Instrumentation(epsilon_p, learning_rate_p, discount_p, n_step_p)
    optimizer = ng.optimizers.NGOpt(parametrization=instru, budget=10)
    recommendation = optimizer.minimize(evaluate)
    optimizer.dump("optimizer_dump")
