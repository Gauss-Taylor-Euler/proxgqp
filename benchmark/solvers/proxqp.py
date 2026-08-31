import numpy

from const import INFINITY, SMALLEST_RELATIVE_TOLERANCE, UNBOUNDED_ITERATIONS
from reformulate import split_linear, gather_dual


STATUS_NAMES = {
    "PROXQP_SOLVED": "solved",
    "PROXQP_PRIMAL_INFEASIBLE": "primal_infeasible",
    "PROXQP_DUAL_INFEASIBLE": "dual_infeasible",
}


class Proxqp:

    name = "proxqp"
    quadratic = True
    cones = ("Zero", "Nonneg")

    @staticmethod
    def solve(problem, eps_abs, warm_start=None):
        from proxsuite import proxqp as backend

        (equality_matrix, equality_value, inequality_matrix, inequality_value,
         equality_rows, inequality_rows) = split_linear(problem)
        variable_count = problem["P"].shape[0]

        solver = backend.sparse.QP(variable_count, equality_matrix.shape[0],
                                   inequality_matrix.shape[0])
        solver.settings.eps_abs = eps_abs
        solver.settings.eps_rel = SMALLEST_RELATIVE_TOLERANCE
        solver.settings.check_duality_gap = True
        solver.settings.eps_primal_inf = eps_abs
        solver.settings.eps_dual_inf = eps_abs
        solver.settings.eps_duality_gap_abs = eps_abs
        solver.settings.eps_duality_gap_rel = SMALLEST_RELATIVE_TOLERANCE
        solver.settings.max_iter = UNBOUNDED_ITERATIONS
        solver.settings.max_iter_in = UNBOUNDED_ITERATIONS
        solver.settings.verbose = False
        solver.init(problem["P"].tocsc(), numpy.asarray(problem["q"], float),
                    equality_matrix, equality_value,
                    inequality_matrix,
                    numpy.full(inequality_value.size, -INFINITY), inequality_value)
        if warm_start is None:
            solver.solve()
        else:
            solver.settings.initial_guess = (
                backend.InitialGuess.WARM_START_WITH_PREVIOUS_RESULT)
            solver.solve(numpy.asarray(warm_start["x"], float),
                         numpy.asarray(warm_start["z"], float)[equality_rows],
                         numpy.asarray(warm_start["z"], float)[inequality_rows])

        status = STATUS_NAMES.get(str(solver.results.info.status).split(".")[-1], "max_iter")
        dual = gather_dual(problem["E"].shape[0], equality_rows, solver.results.y,
                           inequality_rows, solver.results.z)
        return {"status": status,
                "inner": getattr(solver.results.info, "iter", None),
                "outer": getattr(solver.results.info, "iter_ext", None),
                "x": numpy.asarray(solver.results.x, float),
                "z": dual}
