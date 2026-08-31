import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "utils"))

import numpy

from identifier import decode, encode
from portfolio import FORMULATIONS, build

SIZES = ((20, 250), (50, 250), (100, 500), (200, 500), (500, 1000))
FACTOR_COUNTS = (3, 10)
SEEDS = (0, 1, 2)


class SkfolioRandom:

    name = "skfolio_random"
    configurations = tuple(
        (("f", formulation), ("a", assets), ("t", observations),
         ("k", factors), ("s", seed))
        for formulation in range(len(FORMULATIONS))
        for assets, observations in SIZES
        for factors in FACTOR_COUNTS
        for seed in SEEDS)

    @classmethod
    def list_problems(cls):
        return [encode(fields) for fields in cls.configurations]

    @classmethod
    def load(cls, identifier):
        parameters = decode(identifier)
        generator = numpy.random.default_rng(parameters["s"])
        assets, observations = parameters["a"], parameters["t"]
        exposures = generator.normal(size=(observations, parameters["k"]))
        loadings = generator.normal(size=(parameters["k"], assets))
        noise = generator.normal(size=(observations, assets))
        returns = (exposures @ loadings + 0.5 * noise) * 0.01
        return build(FORMULATIONS[parameters["b"]], returns)
