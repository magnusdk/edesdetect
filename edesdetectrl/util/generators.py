import queue
from concurrent.futures._base import Executor
from typing import Any, Callable, Generator


def async_buffered(
    task_executor: Executor,
    buffer_maxsize: int,
    task_gen: Generator[Callable[[], Any], None, None],
):
    """Return a generator that is backed asynchronously by a buffer (queue).

    Guarantees well-behaved (deterministic) random number generation (assuming seed is set).

    Parameters
    ----------
    task_executor : concurrent.futures.Executor
        The task executor, for example an ThreadPoolExecutor. Don't call next on the generator after shutting down/exiting the Executor.
    buffer_maxsize : int
        Maximum size of the buffer. Duh.
    task_gen : A function-returning generator
        A generator that returns a function (task) that will be run asynchronously. `next` is called synchronously on the generator, ensuring that no race conditions can occur.
    """
    # Using a Queue of blocking Futures for a buffer ensures that items are retrieved in the same order that they are inserted.
    buffer = queue.Queue(maxsize=buffer_maxsize)

    # Just a local helper function
    def async_put_next():
        task = next(task_gen)
        future = task_executor.submit(task)
        buffer.put(future)

    # Backfill buffer (asynchronously).
    while not buffer.full():
        async_put_next()

    while True:
        # Get the next item from the buffer.
        # Items are wrapped as Futures and we must block using future.result() to get the value.
        item = buffer.get().result()

        # Every time an item is retrieved from the buffer, add a new one asynchronously.
        async_put_next()

        # Yield the next item from the queue.
        yield item
