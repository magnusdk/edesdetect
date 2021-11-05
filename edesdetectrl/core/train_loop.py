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
    total_steps = 0
    for episode in range(start_episode + 1, num_episodes + 1):
        num_steps = train_episode(training_env, actor)
        total_steps += num_steps
        if episode % 50 == 0:
            print(f"Episode: {episode}, Steps: {total_steps}")
        if episode_stepper:
            episode_stepper.step(episode, total_steps)