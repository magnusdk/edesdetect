from concurrent.futures.thread import ThreadPoolExecutor

import coax
import gym
import optax

import edesdetectrl.dataloaders.echonet as echonet
import edesdetectrl.environments.binary_classification
import edesdetectrl.model as model
from edesdetectrl.config import config
from edesdetectrl.util.checkpoints import CheckpointManager

# FROM: https://coax.readthedocs.io/en/latest/examples/stubs/dqn.html


# pick environment
volumetracings_csv_file = config["data"]["volumetracings_path"]
filelist_csv_file = config["data"]["filelist_path"]
videos_dir = config["data"]["videos_path"]
split = "TRAIN"
thread_pool_executor = ThreadPoolExecutor()
seq_iterator = echonet.get_generator(
    thread_pool_executor,
    volumetracings_csv_file,
    filelist_csv_file,
    videos_dir,
    split,
    buffer_maxsize=50,
)
env = gym.make("EDESClassification-v0", seq_iterator=seq_iterator)
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
buffer = coax.experience_replay.SimpleReplayBuffer(capacity=10000)


# schedule for pi.epsilon (exploration)
epsilon = coax.utils.StepwiseLinearFunction((0, 1), (1000000, 0.1), (2000000, 0.01))


# set up checkpointing and possibly restore a previous checkpoint
checkpoint_path = "_checkpoints/checkpoint.pkl.lz4"
checkpoint_manager = CheckpointManager(q, env, checkpoint_path)
checkpoint_manager.restore_latest()


while env.T < 500:  # 3000000:
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

        # periodically save checkpoints
        if env.ep % 100 == 0:
            checkpoint_manager.save_checkpoint()

        if done:
            break

        s = s_next

    print(f"{env.T}: {env.avg_G}")

trained_params = q.params
coax.utils.dump(trained_params, config["data"]["trained_params_path"])

env.close()
