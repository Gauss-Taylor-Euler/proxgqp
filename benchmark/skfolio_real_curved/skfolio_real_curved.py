import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "utils"))

from identifier import decode, encode
from market import DATASETS, returns_window
from portfolio import FORMULATIONS, build

WINDOWS = (250, 500, 1000)
ERAS = (0, 33, 66, 99)
ASSET_CAP = 200
CHOSEN = (1, 3, 4, 5)


class SkfolioRealCurved:

    name = "skfolio_real_curved"
    configurations = tuple(
        (("f", formulation), ("d", dataset), ("w", window), ("e", era))
        for formulation in CHOSEN
        for dataset in range(len(DATASETS))
        for window in WINDOWS
        for era in ERAS)

    @classmethod
    def list_problems(cls):
        return [encode(fields) for fields in cls.configurations]

    @classmethod
    def load(cls, identifier):
        parameters = decode(identifier)
        returns = returns_window(parameters["d"], parameters["e"] / 100.0,
                                 parameters["w"], ASSET_CAP)
        return build(FORMULATIONS[parameters["b"]], returns)
