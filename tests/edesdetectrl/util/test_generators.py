import random
import time
from concurrent.futures.thread import ThreadPoolExecutor

import edesdetectrl.util.generators as gen


def return_after_random_sleep_task(x):
    def task():
        time.sleep(random.random() / 4)
        return x

    return task


def return_after_sleep_task(x, sleep_for):
    def task():
        time.sleep(sleep_for)
        return x

    return task


def test_async_buffered_no_race_conditions():
    # Should return the same as range(100), i.e: incrementing integers, if every task is run sequentially.
    task_gen = (return_after_random_sleep_task(i) for i in range(100))

    with ThreadPoolExecutor(max_workers=5) as thread_pool_executor:
        bg = gen.async_buffered(task_gen, thread_pool_executor, 5)
        # Taking 10 from this generator should be the same as range(10).
        assert [next(bg) for _ in range(10)] == list(range(10))


def test_async_buffered_runs_concurrently():
    sleep_for = 0.1
    num_workers = 5
    num_tasks_run = 10
    expected_time_spent = sleep_for * (num_tasks_run / num_workers)
    epsilon = 0.01

    task_gen = (return_after_sleep_task(i, sleep_for) for i in range(100))

    with ThreadPoolExecutor(max_workers=num_workers) as thread_pool_executor:
        bg = gen.async_buffered(task_gen, thread_pool_executor, 5)
        t_before = time.time()
        res = [next(bg) for _ in range(num_tasks_run)]
        t_after = time.time()
        delta_t = t_after - t_before

        assert res == list(range(num_tasks_run))
        assert (delta_t - expected_time_spent) < epsilon
