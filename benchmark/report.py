import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils"))

from statistics import (head_to_head, sgm_inner, sgm_seconds, solved_count,
                        succeeded)

RESULT_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "result")


def load(set_name, solver_name):
    path = os.path.join(RESULT_DIRECTORY, "%s_%s.json" % (solver_name, set_name))
    if not os.path.exists(path):
        return None
    return json.load(open(path))


def report(set_name, solver_names, time_limit=60.0, iteration_cap=4000):
    rows = {}
    for solver_name in solver_names:
        loaded = load(set_name, solver_name)
        if loaded is not None:
            rows[solver_name] = loaded
    if not rows:
        print("== %s: no results" % set_name)
        return
    shared = set.intersection(*[{row["name"] for row in r} for r in rows.values()])
    print("== %s   %d problems in common" % (set_name, len(shared)))
    print("   %-20s %10s %12s %12s" % ("solver", "verified", "SGM seconds",
                                       "SGM inner"))
    for solver_name, loaded in rows.items():
        common = [row for row in loaded if row["name"] in shared]
        good, total = solved_count(common)
        seconds = sgm_seconds(common, time_limit)
        counts = [row for row in common if row.get("inner") is not None]
        inner = sgm_inner(common, iteration_cap) if counts else float("nan")
        print("   %-20s %6d/%-3d %12.4f %12s"
              % (solver_name, good, total, seconds,
                 "%.1f" % inner if inner == inner else "n/a"))
    ours = [name for name in rows if name.startswith("proxgqp")]
    others = [name for name in rows if not name.startswith("proxgqp")]
    if ours and others:
        print("   head to head, by iterations where both report them:")
        for mine in ours:
            for theirs in others:
                won, lost = head_to_head(rows[mine], rows[theirs])
                print("     %-20s %3d - %-3d %s" % (mine, won, lost, theirs))


if __name__ == "__main__":
    sets = sys.argv[1].split(",")
    solvers = sys.argv[2].split(",")
    limit = float(sys.argv[3]) if len(sys.argv) > 3 else 60.0
    for name in sets:
        report(name, solvers, limit)
        print()
