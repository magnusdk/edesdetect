from abc import ABC, abstractmethod

from acme import core


class Evaluatable(ABC):
    """Abstract class for getting an actor suitable for evaluation."""
    @abstractmethod
    def get_evaluation_actor(self) -> core.Actor:
        pass
