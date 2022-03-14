import haiku as hk
import jax.numpy as jnp
import jax


def net(S):
    return hk.nets.MobileNetV1(num_classes=2)(S, True)


f = hk.without_apply_rng(hk.transform_with_state(net))
init, apply = f
rng = jax.random.PRNGKey(42)
x = jnp.zeros((256, 7, 112, 112))
# params, state = init(rng, x)

#print(
#    hk.experimental.tabulate(
#        f,
#        columns=["module", "params_bytes"],
#        filters=["has_params"],
#    )(x)
#)
from haiku._src import utils

a = 0
for i in hk.experimental.eval_summary(f)(x):
    a += utils.tree_bytes(i.module_details.params)
    print(utils.tree_bytes(i.module_details.params), "bytes")

print("Total:", a)


# print(apply(params, state, jnp.ones((1, 3, 5, 5)))[0])
