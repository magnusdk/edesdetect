"""Singleton *PoolExecutors."""

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

thread_pool = ThreadPoolExecutor(max_workers=10)
process_pool = ProcessPoolExecutor(max_workers=10)
