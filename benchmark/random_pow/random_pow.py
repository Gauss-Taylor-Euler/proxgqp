import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "utils"))

from random_set import RandomSet


class RandomPow(RandomSet):

    name = "random_pow"
    configurations = tuple(
        (("n", size), ("b", block_count), ("a", alpha_percent), ("s", seed))
        for size, block_count in ((30, 5), (100, 20), (300, 50), (1000, 200))
        for alpha_percent in (10, 30, 50, 70, 90)
        for seed in (0, 1, 2))

    @classmethod
    def cone_specification(cls, parameters):
        size = parameters["n"]
        alpha = parameters["a"] / 100.0
        return ([("Zero", size // 4, {}), ("Nonneg", size // 2, {})]
                + [("Power", 3, {"alpha": alpha})] * parameters["b"])
