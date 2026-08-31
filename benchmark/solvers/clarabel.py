import os

import numpy
import scipy.sparse

from const import (OUT_OF_THE_BOX, SMALLEST_RELATIVE_TOLERANCE,
                   UNBOUNDED_ITERATIONS)


STATUS_NAMES = {
    "Solved": "solved",
    "AlmostSolved": "solved",
    "PrimalInfeasible": "primal_infeasible",
    "AlmostPrimalInfeasible": "primal_infeasible",
    "DualInfeasible": "dual_infeasible",
    "AlmostDualInfeasible": "dual_infeasible",
}


def lower_to_upper_permutation(side):
    permutation = numpy.empty(side * (side + 1) // 2, dtype=numpy.int64)
    position = 0
    for column in range(side):
        for row in range(column, side):
            permutation[position] = row * (row + 1) // 2 + column
            position += 1
    return permutation


class Clarabel:

    name = "clarabel"
    quadratic = True
    cones = ("Zero", "Nonneg", "SecondOrder", "PSDTriangle", "Exponential", "Power")

    @staticmethod
    def solve(problem, eps_abs, warm_start=None):
        import clarabel as backend

        E = problem["E"].tocsc()
        constraint_offset = numpy.asarray(problem["b"], float)
        row_order = numpy.arange(E.shape[0])
        cone_objects = []
        offset = 0
        for kind, size, params in problem["cones"]:
            if kind == "Zero":
                cone_objects.append(backend.ZeroConeT(size))
                offset += size
            elif kind == "Nonneg":
                cone_objects.append(backend.NonnegativeConeT(size))
                offset += size
            elif kind == "SecondOrder":
                cone_objects.append(backend.SecondOrderConeT(size))
                offset += size
            elif kind == "Exponential":
                cone_objects.append(backend.ExponentialConeT())
                offset += 3
            elif kind == "Power":
                cone_objects.append(backend.PowerConeT(params["alpha"]))
                offset += 3
            elif kind == "PSDTriangle":
                rows = size * (size + 1) // 2
                row_order[offset:offset + rows] = (
                    offset + numpy.argsort(lower_to_upper_permutation(size)))
                cone_objects.append(backend.PSDTriangleConeT(size))
                offset += rows
            else:
                raise ValueError("unsupported cone %r" % kind)

        reordered = not numpy.array_equal(row_order, numpy.arange(E.shape[0]))
        if reordered:
            E = E[row_order].tocsc()
            constraint_offset = constraint_offset[row_order]

        settings = backend.DefaultSettings()
        settings.verbose = False
        settings.max_threads = int(os.environ.get("OMP_NUM_THREADS", "1"))
        if not OUT_OF_THE_BOX:
            settings.max_iter = UNBOUNDED_ITERATIONS
            settings.tol_gap_abs = eps_abs
            settings.tol_gap_rel = SMALLEST_RELATIVE_TOLERANCE
            settings.tol_feas = eps_abs
            settings.tol_infeas_abs = eps_abs
            settings.tol_infeas_rel = SMALLEST_RELATIVE_TOLERANCE

        solver = backend.DefaultSolver(problem["P"].tocsc(),
                                       numpy.asarray(problem["q"], float),
                                       E, constraint_offset, cone_objects, settings)
        solution = solver.solve()

        status = STATUS_NAMES.get(str(solution.status), "max_iter")
        x = numpy.asarray(solution.x, float)
        z = numpy.asarray(solution.z, float)
        if reordered:
            restored = numpy.empty_like(z)
            restored[row_order] = z
            z = restored
        return {"status": status, "x": x, "z": z,
                "inner": getattr(solution, "iterations", None)}
