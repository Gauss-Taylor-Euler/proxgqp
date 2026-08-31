import numpy

from const import (INFINITY, OUT_OF_THE_BOX, SMALLEST_RELATIVE_TOLERANCE,
                   UNBOUNDED_ITERATIONS)
from reformulate import split_linear, gather_dual


STATUS_NAMES = {
    "PIQP_SOLVED": "solved",
    "PIQP_PRIMAL_INFEASIBLE": "primal_infeasible",
    "PIQP_DUAL_INFEASIBLE": "dual_infeasible",
}


class Piqp:

    name = "piqp"
    quadratic = True
    cones = ("Zero", "Nonneg")

    @staticmethod
    def solve(problem, eps_abs, warm_start=None):
        import piqp as backend

        (equality_matrix, equality_value, inequality_matrix, inequality_value,
         equality_rows, inequality_rows) = split_linear(problem)
        variable_count = problem["P"].shape[0]
        inequality_count = inequality_value.size

        solver = backend.SparseSolver()
        if not OUT_OF_THE_BOX:
            solver.settings.eps_abs = eps_abs
            solver.settings.eps_rel = SMALLEST_RELATIVE_TOLERANCE
            solver.settings.check_duality_gap = True
            solver.settings.eps_duality_gap_abs = eps_abs
            solver.settings.eps_duality_gap_rel = SMALLEST_RELATIVE_TOLERANCE
            solver.settings.max_iter = UNBOUNDED_ITERATIONS
        solver.settings.verbose = False
        solver.setup(problem["P"].tocsc(), numpy.asarray(problem["q"], float),
                     equality_matrix, equality_value,
                     inequality_matrix,
                     numpy.full(inequality_count, -INFINITY), inequality_value,
                     numpy.full(variable_count, -INFINITY),
                     numpy.full(variable_count, INFINITY))
        solver.solve()

        status = STATUS_NAMES.get(str(solver.result.info.status).split(".")[-1], "max_iter")
        inequality_dual = (numpy.asarray(solver.result.z_u, float)
                           - numpy.asarray(solver.result.z_l, float))
        dual = gather_dual(problem["E"].shape[0], equality_rows, solver.result.y,
                           inequality_rows, inequality_dual)
        return {"status": status,
                "inner": getattr(solver.result.info, "iter", None),
                "x": numpy.asarray(solver.result.x, float),
                "z": dual}
