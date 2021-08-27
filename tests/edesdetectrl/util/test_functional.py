import edesdetectrl.util.functional as func
import operator as op


def test_chainf_functions():
    # `chainf` is a more readable way of chaining multiple function calls.
    v = 1
    f1 = lambda x: x + 3
    f2 = lambda x: x ** 2
    f3 = lambda x: x / 2
    res = func.chainf(v, f1, f2, f3)
    # This is the same as writing `f3(f2(f1(v)))`.
    assert res == 8.0


def test_chainf_sexprs():
    # `chainf` also works with Lisp-like S-expressions.
    # The values will be inserted as the first arguments to the S-expression functions with subsequent arguments coming last.
    # For inserting the values at the end instead (for example when chaining generators), see `chainl`.
    v = 1
    f1 = (op.add, 3)  # S-expressions can be tuples
    f2 = [op.pow, 2]  # ...or lists.
    f3 = lambda x: x / 2  # But we can also mix in regular functions.
    res = func.chainf(v, f1, f2, f3)
    # This is the same as writing `f3(op.pow(op.add(v, 3), 2))`.
    assert res == 8.0


def test_chainl_sexprs():
    # `chainl` is meant to chain generators together.
    # It is very similar to `chainf`, but values will be inserted at the end of S-expressions instead of as the first argument.
    g = range(5)
    t1 = (map, lambda x: x * 3)  # S-expressions can be tuples
    t2 = [filter, lambda x: x % 2 == 0]  # ...or lists.
    res = func.chainl(g, t1, t2)
    # This is the same as writing `filter(lambda x: x % 2 == 0, map(lambda x: x * 3, range(10)))`.
    assert list(res) == [0, 6, 12]
    # Semantically, the above statements says:
    # 1. Take the sequence of numbers [0,5) (= [0,1,2,3,4])
    # 2. Multiply each number by 3 (= [0,3,6,9,12])
    # 3. Filter out all the odd numbers (= [0, 6, 12])
    # 4. Assert that the result equals [0, 6, 12].

    # Here's the same example in a (possibly) more readable format:
    assert (
        func.chainl(
            range(5),
            (map, lambda x: x * 3),
            (filter, lambda x: x % 2 == 0),
            list,
        )
        == [0, 6, 12]
    )
