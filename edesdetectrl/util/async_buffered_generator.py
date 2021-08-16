import queue


def get_async_buffered_generator(task_executor, get_task_fn, buffer_maxsize=10):
    """Return a generator that is backed asynchronously by a buffer (queue).

    Guarantees well-behaved (deterministic) random number generation (assuming seed is set).

    Parameters
    ----------
    task_executor : concurrent.futures.Executor
        The task executor, for example an ThreadPoolExecutor. Don't call next on the generator after shutting down/exiting the Executor.
    get_task_fn : A function that returns a task_fn
        A second-order function returns the function that will be run in another thread. get_task_fn is guaranteed to run synchronously, so any impure function call (for example random number generation) should be done outside the task_fn closure, i.e. it should be done in get_task_fn.
    buffer_maxsize : int
        Maximum size of the buffer. Duh.
    """
    # Using a Queue of blocking Futures for a buffer ensures that items are retrieved in the same order that they are inserted.
    buffer = queue.Queue(maxsize=buffer_maxsize)

    # Just a local helper function
    def async_put_next():
        task_fn = get_task_fn()
        future = task_executor.submit(task_fn)
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
