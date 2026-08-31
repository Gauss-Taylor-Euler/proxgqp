from highs_model import run


class HighsSimplex:

    name = "highs_simplex"
    quadratic = True
    cones = ("Zero", "Nonneg")

    @staticmethod
    def solve(problem, eps_abs, warm_start=None):
        return run(problem, eps_abs, "simplex")
