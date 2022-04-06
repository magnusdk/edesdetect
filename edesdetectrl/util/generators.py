import queue
from concurrent.futures import Executor
from typing import Any, Callable, Generator, Tuple, Union

SKIP_ITEM = "__SKIP_ITEM__"


def async_buffered(
    task_gen: Generator[Union[Tuple[Callable[[Tuple], Any], Tuple], str], None, None],
    task_executor: Executor,
    buffer_maxsize: int,
):
    """Return a generator that is backed asynchronously by a buffer (queue).

    Guarantees well-behaved (deterministic) random number generation (assuming seed is set).

    Parameters
    ----------
    task_gen : A function-returning generator
        A generator that returns a function (task) that will be run asynchronously. `next` is called synchronously on the generator, ensuring that no race conditions can occur.
    task_executor : concurrent.futures.Executor
        The task executor, for example an ThreadPoolExecutor. Don't call next on the generator after shutting down/exiting the Executor.
    buffer_maxsize : int
        Maximum size of the buffer. Duh.
    """
    # Using a Queue of blocking Futures for a buffer ensures that items are retrieved in the same order that they are inserted.
    buffer = queue.Queue(maxsize=buffer_maxsize)

    # Just a local helper function
    def async_put_next():
        next_item = next(task_gen)
        while next_item == SKIP_ITEM:
            next_item = next(task_gen)

        if next_item != SKIP_ITEM:
            task_fn, args = next_item
            future = task_executor.submit(task_fn, *args)
            buffer.put(future)

    # Backfill buffer (asynchronously).
    while not buffer.full():
        async_put_next()

    while True:
        # Get the next item from the buffer.
        # Items are wrapped as Futures and we must block using future.result() to get the value.
        item = buffer.get().result()

        # Skip items marked for skipping.
        while item == SKIP_ITEM:
            item = buffer.get().result()

        # Every time an item is retrieved from the buffer, add a new one asynchronously.
        async_put_next()

        # Yield the next item from the queue.
        yield item
