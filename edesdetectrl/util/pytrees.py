import dataclasses

import jax


def dataclass_as_shallow_dict(obj):
    return {a.name: getattr(obj, a.name) for a in dataclasses.fields(obj)}


def register_pytree_node_dataclass(cls, *args, **kwargs):
    cls = dataclasses.dataclass(cls, *args, **kwargs)
    _flatten = lambda obj: jax.tree_flatten(dataclass_as_shallow_dict(obj))
    _unflatten = lambda d, children: cls(**d.unflatten(children))
    jax.tree_util.register_pytree_node(cls, _flatten, _unflatten)
    return cls
