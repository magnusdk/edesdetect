from acme import core
import edesdetectrl.acme_agents.extensions as extensions


class AlwaysPicksActionActor(core.Actor):
    """An Actor that only ever picks the given action."""
    def __init__(self, action):
        self.action = action
        super().__init__()

    def select_action(self, observation):
        return self.action

    def observe_first(self, timestep):
        pass

    def observe(self, action, next_timestep):
        pass

    def update(self, wait=False):
        pass


class AlwaysPicksActionEvaluatable(extensions.Evaluatable):
    def __init__(self, action):
        self.action = action
        super().__init__()

    def get_evaluation_actor(self):
        return AlwaysPicksActionActor(self.action)
