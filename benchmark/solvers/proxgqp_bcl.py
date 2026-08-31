import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parent))

from proxgqp_interior import solve_with


class ProxGqpBcl:

    name = "proxgqp_bcl"
    quadratic = True
    cones = ("Zero", "Nonneg", "SecondOrder", "PSDTriangle", "Exponential", "Power")

    @staticmethod
    def solve(problem, eps_abs, warm_start=None):
        return solve_with(problem, eps_abs, warm_start, smoothing=1,
                          **{"semismooth.penalty": "bcl"})
