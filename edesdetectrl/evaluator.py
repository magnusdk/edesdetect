import edesdetectrl.acme_agents.extensions as extensions
from edesdetectrl import environments


def avg_metrics(all_metrics: dict):
    result = {}
    for k in all_metrics[0].keys():
        avg = sum(map(lambda m: m[k], all_metrics)) / len(all_metrics)
        result[k] = avg
    return result


class Evaluator:
    def __init__(
        self,
        env,
        evaluatable: extensions.Evaluatable,
        n_trajectories: int,
        delta_episodes: int,
        start_episode: int = 0,
    ):
        self._env = env
        self._evaluatable = evaluatable
        self._n_trajectories = n_trajectories
        self._delta_episodes = delta_episodes
        self._episode = start_episode

    def evaluate(self):
        """Create n_trajectories trajectories and return metrics on them."""
        actor = self._evaluatable.get_evaluation_actor()

        metrics = []
        for _ in range(self._n_trajectories):
            trajectory = environments.generate_trajectory_using_actor(self._env, actor)
            metrics.append(trajectory.all_metrics())

        return avg_metrics(metrics)

    def step(self):
        self._episode += 1
        if self._episode % self._delta_episodes == 0:
            metrics = self.evaluate()
            return metrics
