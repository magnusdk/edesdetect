from edesdetectrl.core.stepper import Stepper, Combined


def test_Combined():
    a = 0  # Incremented differently for each stepper
    b = 0  # Incremented by n (argument to step()) for each stepper

    class Stepper1(Stepper):
        def step(self, n):
            nonlocal a, b
            a += 1
            b += n

    class Stepper2(Stepper):
        def step(self, n):
            nonlocal a, b
            a += 2
            b += n

    combined = Combined(Stepper1(), Stepper2())
    combined.step(3)

    assert a == 3  # Stepper1 added 1 and Stepper2 added 2 to a
    assert b == 6  # Both steppers added 3 to b
