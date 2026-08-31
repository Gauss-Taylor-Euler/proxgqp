import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "utils"))

from random_set import RandomSet


class RandomExp(RandomSet):

    name = "random_exp"
    configurations = tuple(
        (("n", size), ("b", block_count), ("c", condition), ("s", seed))
        for size, block_count in ((30, 5), (100, 20), (300, 50),
                                  (1000, 200), (3000, 500))
        for condition in (1, 3)
        for seed in (0, 1, 2, 3, 4, 5))

    @classmethod
    def cone_specification(cls, parameters):
        size = parameters["n"]
        return ([("Zero", size // 4, {}), ("Nonneg", size // 2, {})]
                + [("Exponential", 3, {})] * parameters["b"])
