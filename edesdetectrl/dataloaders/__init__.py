from jax import random
import edesdetectrl.util.generators as generators
from concurrent.futures import Executor


def _safe_get_item(items, index):
    try:
        return items[index]
    except KeyError:
        return generators.SKIP_ITEM


class DataLoader:
    @property
    def keys(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.keys)

    def get_random_generator(
        self,
        rng_key,
        executor: Executor,
        prefetch: int = 10,
    ):
        """Return a generator that generates all items from dataloader in random order."""

        def task_gen():
            key1 = rng_key
            while True:
                key1, key2 = random.split(key1)
                yield lambda: _safe_get_item(
                    self, random.randint(key2, (1,), 0, len(self))[0]
                )

        generator = task_gen()
        return generators.async_buffered(executor, prefetch, generator)

    def get_generator(
        self,
        executor: Executor,
        prefetch: int = 10,
    ):
        """Return a generator that generates all items from dataloader, ordered by index."""

        def task_gen():
            for index in range(len(self)):
                yield lambda: _safe_get_item(self, index)

        generator = task_gen()
        return generators.async_buffered(executor, prefetch, generator)
