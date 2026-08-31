import pathlib
import sys

import numpy

from const import UNBOUNDED_ITERATIONS, imposed

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
    chosen = imposed(eps_abs=eps_abs, eps_rel=eps_abs,
                     eps_gap_abs=eps_abs, eps_gap_rel=eps_abs,
                     max_iter_outer=UNBOUNDED_ITERATIONS,
                     max_newton=UNBOUNDED_ITERATIONS)
    chosen.update(overrides)
    return proxgqp.make_settings(method=method, **chosen)


def make_problem(problem):
    cones = [BUILDERS[name](size, extra) for name, size, extra in problem["cones"]]
    return Problem(problem["P"], problem["q"], problem["E"], problem["b"], cones)


def run(problem, settings, warm_start=None):
    built = make_problem(problem)
    warm = None
    if warm_start is not None and all(key in warm_start for key in ("x", "z")):
        given = dict(warm_start)
        if "s" not in given:
            given["s"] = numpy.asarray(problem["b"], float).ravel() - \
                problem["E"] @ numpy.asarray(given["x"], float).ravel()
        sizes = (built.columns, built.rows, built.rows)
        parts = [numpy.asarray(given[key]).ravel() for key in ("x", "s", "z")]
        if all(part.size == size for part, size in zip(parts, sizes)):
            warm = given
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
