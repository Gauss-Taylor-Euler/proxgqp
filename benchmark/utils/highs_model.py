import os

import numpy
import scipy.sparse

from const import (INFINITY, REGULARIZATION_BELOW_TOLERANCE,
                   UNBOUNDED_ITERATIONS)
from reformulate import equality_inequality_rows

STATUS_NAMES = {
    "Optimal": "solved",
    "Infeasible": "primal_infeasible",
    "Unbounded": "dual_infeasible",
    "Primal infeasible or unbounded": "primal_infeasible",
}


def build(problem, backend):
    equality_rows, inequality_rows = equality_inequality_rows(problem["cones"])
    f = numpy.asarray(problem["f"], float)
    row_lower = numpy.empty_like(f)
    row_upper = numpy.empty_like(f)
    row_lower[equality_rows] = f[equality_rows]
    row_upper[equality_rows] = f[equality_rows]
    row_lower[inequality_rows] = -INFINITY
    row_upper[inequality_rows] = f[inequality_rows]

    E = problem["E"].tocsc()
    variable_count, row_count = E.shape[1], E.shape[0]

    model = backend.HighsLp()
    model.num_col_ = variable_count
    model.num_row_ = row_count
    model.col_cost_ = numpy.asarray(problem["q"], float)
    model.col_lower_ = numpy.full(variable_count, -INFINITY)
    model.col_upper_ = numpy.full(variable_count, INFINITY)
    model.row_lower_ = row_lower
    model.row_upper_ = row_upper
    model.a_matrix_.format_ = backend.MatrixFormat.kColwise
    model.a_matrix_.start_ = E.indptr.astype(numpy.int32)
    model.a_matrix_.index_ = E.indices.astype(numpy.int32)
    model.a_matrix_.value_ = E.data
    return model


def build_hessian(problem, backend):
    P = scipy.sparse.tril(problem["P"], format="csc")
    if P.nnz == 0:
        return None
    hessian = backend.HighsHessian()
    hessian.dim_ = P.shape[0]
    hessian.format_ = backend.HessianFormat.kTriangular
    hessian.start_ = P.indptr.astype(numpy.int32)
    hessian.index_ = P.indices.astype(numpy.int32)
    hessian.value_ = P.data
    return hessian


def run(problem, eps_abs, method):
    import highspy as backend

    solver = backend.Highs()
    solver.setOptionValue("output_flag", False)
    solver.setOptionValue("solver", method)
    solver.setOptionValue("threads", int(os.environ.get("OMP_NUM_THREADS", "1")))
    solver.setOptionValue("time_limit", float("inf"))
    solver.setOptionValue("simplex_iteration_limit", UNBOUNDED_ITERATIONS)
    solver.setOptionValue("ipm_iteration_limit", UNBOUNDED_ITERATIONS)
    solver.setOptionValue("qp_iteration_limit", UNBOUNDED_ITERATIONS)
    solver.setOptionValue("primal_feasibility_tolerance", eps_abs)
    solver.setOptionValue("dual_feasibility_tolerance", eps_abs)
    solver.setOptionValue("ipm_optimality_tolerance", eps_abs)
    solver.setOptionValue("primal_residual_tolerance", eps_abs)
    solver.setOptionValue("dual_residual_tolerance", eps_abs)
    solver.setOptionValue("optimality_tolerance", eps_abs)
    solver.setOptionValue("kkt_tolerance", eps_abs)
    solver.setOptionValue("qp_regularization_value",
                          eps_abs * REGULARIZATION_BELOW_TOLERANCE)

    solver.passModel(build(problem, backend))
    hessian = build_hessian(problem, backend)
    if hessian is not None:
        solver.passHessian(hessian)
    solver.run()

    status = STATUS_NAMES.get(solver.modelStatusToString(solver.getModelStatus()),
                              "max_iter")
    solution = solver.getSolution()
    primal = numpy.asarray(solution.col_value, float)
    dual = -numpy.asarray(solution.row_dual, float)
    if primal.size != problem["P"].shape[0] or dual.size != problem["E"].shape[0]:
        return {"status": status, "x": None, "z": None}
    return {"status": status, "x": primal, "z": dual}
