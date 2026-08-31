import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "utils"))

from random_set import RandomSet


class RandomSocp(RandomSet):

    name = "random_socp"
    configurations = tuple(
        (("n", size), ("b", block_dimension), ("c", condition), ("s", seed))
        for size in (30, 100, 300, 1000, 3000)
        for block_dimension in (4, 50)
        for condition in (1, 3, 6)
        for seed in (0, 1, 2))

    @classmethod
    def cone_specification(cls, parameters):
        size = parameters["n"]
        block_dimension = parameters["b"]
        block_count = max(1, size // block_dimension)
        return ([("Zero", size // 4, {}), ("Nonneg", size // 2, {})]
                + [("SecondOrder", block_dimension, {})] * block_count)
