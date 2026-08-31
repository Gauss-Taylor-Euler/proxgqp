"""Aggregates over a benchmark run.

A failure is not dropped and not excluded: it is scored at the cap, which is
the time limit for seconds and the iteration budget for counts.  Dropping
failures would reward a solver for giving up early, and excluding them would
compare solvers on different problem sets.
"""
import math

SECONDS_SHIFT = 1e-3
ITERATION_SHIFT = 10.0


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


def sgm_inner(rows, iteration_cap):
    return shifted_geometric_mean(capped(rows, "inner", iteration_cap),
                                  ITERATION_SHIFT)


def solved_count(rows):
    return sum(1 for row in rows if succeeded(row)), len(rows)


def head_to_head(rows_a, rows_b, field="inner"):
    by_a = {row["name"]: row for row in rows_a}
    by_b = {row["name"]: row for row in rows_b}
    wins_a = wins_b = 0
    for name in set(by_a) & set(by_b):
        a, b = by_a[name], by_b[name]
        good_a, good_b = succeeded(a), succeeded(b)
        if good_a and good_b:
            if a.get(field) is None or b.get(field) is None:
                continue
            if a[field] < b[field]:
                wins_a += 1
            elif b[field] < a[field]:
                wins_b += 1
        elif good_a:
            wins_a += 1
        elif good_b:
            wins_b += 1
    return wins_a, wins_b


def easy_hard_split(rows_by_solver, reference, threshold=130):
    names = set()
    for rows in rows_by_solver.values():
        names |= {row["name"] for row in rows}
    by_reference = {row["name"]: row for row in rows_by_solver.get(reference, [])}
    hard = set()
    for name in names:
        row = by_reference.get(name)
        if row is None or not succeeded(row):
            hard.add(name)
        elif row.get("inner") is not None and row["inner"] >= threshold:
            hard.add(name)
    return {"easy": names - hard, "hard": hard}
