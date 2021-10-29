from acme import core


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


class AlwaysPicksActionEvaluatable:
    def __init__(self, action):
        self.action = action
        super().__init__()

    def get_evaluation_actor(self):
        return AlwaysPicksActionActor(self.action)
