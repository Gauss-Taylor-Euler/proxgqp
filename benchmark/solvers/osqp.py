import numpy
import scipy.sparse

from const import INFINITY, SMALLEST_RELATIVE_TOLERANCE, UNBOUNDED_ITERATIONS
from reformulate import equality_inequality_rows


STATUS_NAMES = {
    "solved": "solved",
    "solved inaccurate": "solved",
    "primal infeasible": "primal_infeasible",
    "primal infeasible inaccurate": "primal_infeasible",
    "dual infeasible": "dual_infeasible",
    "dual infeasible inaccurate": "dual_infeasible",
}


class Osqp:

    name = "osqp"
    quadratic = True
    cones = ("Zero", "Nonneg")

    @staticmethod
    def solve(problem, eps_abs, warm_start=None):
        import osqp as backend

        equality_rows, inequality_rows = equality_inequality_rows(problem["cones"])
        f = numpy.asarray(problem["f"], float)
        lower = numpy.empty_like(f)
        upper = numpy.empty_like(f)
        lower[equality_rows] = f[equality_rows]
        upper[equality_rows] = f[equality_rows]
        lower[inequality_rows] = -INFINITY
        upper[inequality_rows] = f[inequality_rows]

        solver = backend.OSQP()
        solver.setup(scipy.sparse.triu(problem["P"], format="csc"),
                     numpy.asarray(problem["q"], float),
                     problem["E"].tocsc(), lower, upper,
                     eps_abs=eps_abs, eps_rel=SMALLEST_RELATIVE_TOLERANCE,
                     eps_prim_inf=eps_abs, eps_dual_inf=eps_abs,
                     max_iter=UNBOUNDED_ITERATIONS, polishing=True, verbose=False)
        if warm_start is not None:
            solver.warm_start(x=numpy.asarray(warm_start["x"], float),
                              y=numpy.asarray(warm_start["z"], float))
        solution = solver.solve()
        status = STATUS_NAMES.get(str(solution.info.status).strip().lower(), "max_iter")
        return {"status": status,
                "inner": getattr(solution.info, "iter", None),
                "x": numpy.asarray(solution.x, float),
                "z": numpy.asarray(solution.y, float)}
