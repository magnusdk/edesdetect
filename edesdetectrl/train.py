import coax
import optax
import gym

import edesdetectrl.environments.binary_classification
from edesdetectrl.config import config
import edesdetectrl.model as model


# FROM: https://coax.readthedocs.io/en/latest/examples/stubs/dqn.html


# pick environment
env = gym.make("EDESClassification-v0")
env = coax.wrappers.TrainMonitor(
    env, tensorboard_dir="tensorboard", tensorboard_write_all=True, log_all_metrics=True
)

# function approximator
q = coax.Q(model.get_func_approx(env), env)
pi = coax.EpsilonGreedy(q, epsilon=0.1)


# target network
q_targ = q.copy()


# specify how to update q-function
qlearning = coax.td_learning.QLearning(q, q_targ=q_targ, optimizer=optax.adam(0.001))


# specify how to trace the transitions
tracer = coax.reward_tracing.NStep(n=1, gamma=0.9)
buffer = coax.experience_replay.SimpleReplayBuffer(capacity=1000000)


# schedule for pi.epsilon (exploration)
epsilon = coax.utils.StepwiseLinearFunction((0, 1), (1000000, 0.1), (2000000, 0.01))


while env.T < 50:  # 3000000:
    pi.epsilon = epsilon(env.T)
    s = env.reset()

    for t in range(env.spec.max_episode_steps):
        a = pi(s)
        s_next, r, done, info = env.step(a)

        # add transition to buffer
        tracer.add(s, a, r, done)
        while tracer:
            transition = tracer.pop()
            buffer.add(transition)

        # update
        if len(buffer) >= 32:
            transition_batch = buffer.sample(batch_size=32)
            metrics = qlearning.update(transition_batch)
            env.record_metrics(metrics)

        # periodically sync target model
        if env.ep % 10 == 0:
            q_targ.soft_update(q, tau=1.0)

        if done:
            break

        s = s_next

    print(f"{env.T}: {env.avg_G}")

trained_params = q.params
coax.utils.dump(trained_params, config['data']['trained_params_path'])

env.close()
