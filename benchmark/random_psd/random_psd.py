import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "utils"))

from random_set import RandomSet


class RandomPsd(RandomSet):

    name = "random_psd"
    configurations = tuple(
        (("k", side), ("b", block_count), ("c", condition), ("s", seed))
        for side in (3, 5, 8, 12, 20)
        for block_count in (1, 4)
        for condition in (1, 3)
        for seed in (0, 1, 2))

    @classmethod
    def variable_count(cls, parameters):
        side = parameters["k"]
        return parameters["b"] * side * (side + 1) // 2

    @classmethod
    def cone_specification(cls, parameters):
        side = parameters["k"]
        variables = cls.variable_count(parameters)
        return ([("Zero", max(1, variables // 8), {})]
                + [("PSDTriangle", side, {})] * parameters["b"])
