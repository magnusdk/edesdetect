import time


class Timer:
    """A simple class for keeping track of when to run periodic tasks.

    check() checks whether time_delta_seconds seconds has passed since the last reset.
    reset() resets the timer.

    Example:

    timer = Timer(10)  # The periodic task should run every 10 seconds
    while True:
        if timer.check():
            periodic_task()
            timer.reset()
    """

    def __init__(self, time_delta_seconds: float):
        """"""
        self._time_delta_seconds = time_delta_seconds
        self.reset()

    def check(self) -> bool:
        """Return true if time_delta_seconds seconds has passed since the last reset()."""
        return time.time() > self._last_reset + self._time_delta_seconds

    def reset(self):
        """Reset the timer. After calling this method, check() will return false until time_delta_seconds seconds has passed."""
        self._last_reset = time.time()
