from highs_model import run


class HighsIpm:

    name = "highs_ipm"
    quadratic = True
    cones = ("Zero", "Nonneg")

    @staticmethod
    def solve(problem, eps_abs, warm_start=None):
        return run(problem, eps_abs, "ipm")
