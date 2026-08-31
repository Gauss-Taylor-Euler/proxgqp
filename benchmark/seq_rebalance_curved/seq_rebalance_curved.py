import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "utils"))

from identifier import decode, encode
from market import returns_window
from portfolio import build

FAMILIES = ("rolling", "frontier", "beta")
STEP_COUNT = 40
DATASET = 1
WINDOW = 500
ASSET_CAP = 60


class SeqRebalanceCurved:

    name = "seq_rebalance_curved"
    configurations = tuple(
        (("g", family), ("i", step))
        for family in range(len(FAMILIES))
        for step in range(STEP_COUNT))

    @classmethod
    def list_problems(cls):
        return [encode(fields) for fields in cls.configurations]

    @classmethod
    def sequence_key(cls, identifier):
        return decode(identifier)["g"]

    @classmethod
    def load(cls, identifier):
        parameters = decode(identifier)
        family = FAMILIES[parameters["g"]]
        step = parameters["i"]
        if family == "rolling":
            returns = returns_window(DATASET, 0.3 + 0.4 * step / STEP_COUNT,
                                     WINDOW, ASSET_CAP)
            return build("min_deviation", returns)
        returns = returns_window(DATASET, 0.5, WINDOW, ASSET_CAP)
        if family == "frontier":
            return build("mean_evar", returns,
                         risk_aversion=10.0 ** (-2.0 + 4.0 * step / STEP_COUNT))
        return build("mean_evar", returns, beta=0.80 + 0.19 * step / STEP_COUNT)
