from functools import reduce


def chainf(initial, *args):
    """Chain functions and pseudo S-expressions like macro-thread-first from languages like Clojure, i.e.: `->`.

    How this works is best understood by looking at examples. For examples see tests.
    """
    return reduce(lambda v, f: f(v) if callable(f) else f[0](v, *f[1:]), args, initial)


def chainl(initial, *args):
    """Chain functions and pseudo S-expressions like macro-thread-last from languages like Clojure, i.e.: `->>`.

    This function should only be used to chain generators.

    How this works is best understood by looking at examples. For examples see tests."""
    return reduce(lambda v, f: f(v) if callable(f) else f[0](*f[1:], v), args, initial)
