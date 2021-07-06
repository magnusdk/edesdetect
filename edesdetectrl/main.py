import coax
import optax
import haiku as hk
import jax
import jax.numpy as jnp
import gym

import edesdetectrl.environments.binary_classification


# FROM: https://coax.readthedocs.io/en/latest/examples/stubs/dqn.html


# pick environment
env = gym.make("EDESClassification-v0")
env = coax.wrappers.TrainMonitor(
    env, tensorboard_dir="tensorboard", tensorboard_write_all=True
)


def func_approx(S, is_training):
    f = hk.Sequential(
        [
            coax.utils.diff_transform,  # Preprocess the frames to get position, velocity, acceleration, etc...
            hk.Conv2D(16, kernel_shape=8, stride=4),
            jax.nn.relu,
            hk.Conv2D(32, kernel_shape=4, stride=2),
            jax.nn.relu,
            hk.Flatten(),
            hk.Linear(256),
            jax.nn.relu,
            hk.Linear(env.action_space.n, w_init=jnp.zeros),
        ]
    )
    return f(S)  # Output shape: (batch_size, num_actions=2)


# function approximator
q = coax.Q(func_approx, env)
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


while env.T < 5000:  # 3000000:
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

env.close()
