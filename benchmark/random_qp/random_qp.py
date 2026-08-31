import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "utils"))

from random_set import RandomSet


class RandomQp(RandomSet):

    name = "random_qp"
    configurations = tuple(
        (("n", size), ("c", condition), ("r", rank), ("s", seed))
        for size in (30, 100, 300, 1000, 3000)
        for condition in (1, 3, 6)
        for rank in (-1, 0)
        for seed in (0, 1, 2))

    @classmethod
    def cone_specification(cls, parameters):
        size = parameters["n"]
        return [("Zero", size // 2, {}), ("Nonneg", size, {})]
