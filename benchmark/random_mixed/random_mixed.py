import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "utils"))

from random_set import RandomSet


class RandomMixed(RandomSet):

    name = "random_mixed"
    configurations = tuple(
        (("n", size), ("c", condition), ("s", seed))
        for size in (30, 100, 300, 1000, 3000)
        for condition in (1, 3, 6)
        for seed in (0, 1, 2, 3))

    @classmethod
    def cone_specification(cls, parameters):
        size = parameters["n"]
        block_count = max(1, size // 30)
        second_order_dimension = max(3, size // 20)
        side = max(2, min(12, size // 40))
        return ([("Zero", size // 8, {}), ("Nonneg", size // 4, {})]
                + [("SecondOrder", second_order_dimension, {})] * block_count
                + [("PSDTriangle", side, {})]
                + [("Exponential", 3, {})] * block_count
                + [("Power", 3, {"alpha": 0.35})] * block_count)
