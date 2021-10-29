from abc import ABC, abstractmethod
from typing import List


class Stepper(ABC):
    @abstractmethod
    def step(*args):
        pass


class Combined(Stepper):
    def __init__(self, *steppers: List[Stepper]):
        self.steppers = steppers

    def step(self, *args):
        for stepper in self.steppers:
            stepper.step(*args)
