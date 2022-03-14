import jax.numpy as jnp
from jax import tree_util

"""Some helpers for the sharp edges of JAX."""


def where(cond, x, y):
    """Same as jnp.where, but accept custom pytree nodes.

    x and y must have the same treedef when flattened."""
    x_leaves, treedef = tree_util.tree_flatten(x)
    y_leaves, _ = tree_util.tree_flatten(y)
    result = jnp.where(cond, jnp.array(x_leaves), jnp.array(y_leaves))
    return tree_util.tree_unflatten(treedef, result)
