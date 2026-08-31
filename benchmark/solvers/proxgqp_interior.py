import pathlib
import sys

import numpy

PACKAGE_DIR = pathlib.Path(__file__).resolve().parents[2] / "solver" / "python"
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

import proxgqp
from proxgqp import (Exponential, Nonneg, PSDTriangle, Power, Problem,
                     SecondOrder, Zero)

STATUS = proxgqp.STATUS

BUILDERS = {"Zero": lambda size, extra: Zero(size),
            "Nonneg": lambda size, extra: Nonneg(size),
            "SecondOrder": lambda size, extra: SecondOrder(size),
            "PSDTriangle": lambda size, extra: PSDTriangle(size),
            "Exponential": lambda size, extra: Exponential(),
            "Power": lambda size, extra: Power(extra.get("alpha", 0.5))}


def make_settings(eps_abs, smoothing, **overrides):
    method = "semismooth" if smoothing == 1 else "interior"
    return proxgqp.make_settings(eps_abs=eps_abs, eps_rel=eps_abs,
                                 eps_gap_abs=eps_abs, eps_gap_rel=eps_abs,
                                 method=method, **overrides)


def make_problem(problem):
    cones = [BUILDERS[name](size, extra) for name, size, extra in problem["cones"]]
    return Problem(problem["P"], problem["q"], problem["E"], problem["f"], cones)


def run(problem, settings, warm_start=None):
    built = make_problem(problem)
    warm = None
    if warm_start is not None and all(key in warm_start for key in ("x", "s", "z")):
        sizes = (built.columns, built.rows, built.rows)
        parts = [numpy.asarray(warm_start[key]).ravel() for key in ("x", "s", "z")]
        if all(part.size == size for part, size in zip(parts, sizes)):
            warm = warm_start
    return proxgqp.solve(built, warm_start=warm, settings=settings)


def solve_with(problem, eps_abs, warm_start, smoothing, **overrides):
    solution = run(problem, make_settings(eps_abs, smoothing, **overrides),
                   warm_start)
    return {"status": solution.status, "x": solution.x, "s": solution.s,
            "z": solution.z, "outer": solution.outer_iterations,
            "inner": solution.inner_iterations}


class ProxGqpInterior:

    name = "proxgqp_interior"
    quadratic = True
    cones = ("Zero", "Nonneg", "SecondOrder", "PSDTriangle", "Exponential", "Power")

    @staticmethod
    def solve(problem, eps_abs, warm_start=None):
        return solve_with(problem, eps_abs, warm_start, smoothing=0)
