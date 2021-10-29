import dm_env
from acme import core
from edesdetectrl.core.stepper import Stepper


def train_episode(
    env: dm_env.Environment,
    actor: core.Actor,
) -> int:
    timestep = env.reset()
    num_steps = 1
    actor.observe_first(timestep)
    while not timestep.last():
        action = actor.select_action(timestep.observation)
        timestep = env.step(action)
        num_steps += 1
        actor.observe(action, next_timestep=timestep)
        actor.update()

    return num_steps


def train_loop(
    training_env: dm_env.Environment,
    actor: core.Actor,
    num_episodes: int,
    start_episode: int = 0,
    episode_stepper: Stepper = None,
):
    for episode in range(start_episode+1, num_episodes+1):
        num_steps = train_episode(training_env, actor)
        if episode_stepper:
            episode_stepper.step(episode, num_steps)