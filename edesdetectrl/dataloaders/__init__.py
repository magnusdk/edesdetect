from itertools import count

import edesdetectrl.util.generators as generators
from jax import random


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

    def get_random_generator(self, rng_key, as_task=True):
        """Return a generator that generates all items from dataloader in random order.

        If as_task is True, then generated values are 0-arity functions that can be
        called to get the actual value. This is useful when wrapping the generator in
        an asynchronous generator."""
        key1 = rng_key
        while True:
            key1, key2 = random.split(key1)
            task = lambda: _safe_get_item(
                self, random.randint(key2, (1,), 0, len(self))[0]
            )
            yield task if as_task else task()

    def get_generator(self, as_task=True):
        """Return a generator that generates all items from dataloader, ordered by index.

        The generator will generate indefinitely, cycling back to the start after all items have been generated.

        If as_task is True, then generated values are 0-arity functions that can be
        called to get the actual value. This is useful when wrapping the generator in
        an asynchronous generator."""
        for n in count():
            index = n % len(self)
            task = lambda: _safe_get_item(self, index)
            yield task if as_task else task()
