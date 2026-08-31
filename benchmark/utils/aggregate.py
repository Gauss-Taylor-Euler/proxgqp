"""Aggregates over a benchmark run.

A failure is not dropped and not excluded: it is scored at the cap, which is
the time limit.  Dropping failures would reward a solver for giving up early,
and excluding them would compare solvers on different problem sets.
"""
import math

SECONDS_SHIFT = 1e-3


def shifted_geometric_mean(values, shift):
    if not values:
        return float("nan")
    total = sum(math.log(max(value, 0.0) + shift) for value in values)
    return math.exp(total / len(values)) - shift


def succeeded(row):
    return bool(row.get("verified")) and row.get("status") == "solved"


def capped(rows, field, cap):
    out = []
    for row in rows:
        value = row.get(field)
        out.append(value if succeeded(row) and value is not None else cap)
    return out


def sgm_seconds(rows, time_limit):
    return shifted_geometric_mean(capped(rows, "seconds", time_limit),
                                  SECONDS_SHIFT)


def solved_count(rows):
    return sum(1 for row in rows if succeeded(row)), len(rows)
