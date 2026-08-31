import numpy

from const import (SMALLEST_RELATIVE_TOLERANCE, UNBOUNDED_ITERATIONS,
                   imposed)
from reformulate import rows_by_kind

INEQUALITY_ORDER = ("Nonneg", "SecondOrder", "Exponential")
EXPONENTIAL_TO_ECOS = [0, 2, 1]

STATUS_NAMES = {
    0: "solved",
    10: "solved",
    1: "primal_infeasible",
    11: "primal_infeasible",
    2: "dual_infeasible",
    12: "dual_infeasible",
}


def concatenate_rows(pieces):
    if not pieces:
        return numpy.zeros(0, dtype=numpy.int64)
    return numpy.concatenate(pieces)


class Ecos:

    name = "ecos"
    quadratic = False
    cones = ("Zero", "Nonneg", "SecondOrder", "Exponential")

    @staticmethod
    def solve(problem, eps_abs, warm_start=None):
        import ecos as backend

        if problem["P"].nnz > 0:
            raise ValueError("ecos has no quadratic objective")

        blocks = rows_by_kind(problem["cones"])
        unsupported = set(blocks) - set(Ecos.cones)
        if unsupported:
            raise ValueError("ecos cannot take %s" % sorted(unsupported))

        equality_rows = concatenate_rows(
            [rows for _size, _params, rows in blocks.get("Zero", [])])
        inequality_pieces = []
        for kind in INEQUALITY_ORDER:
            for _size, _params, rows in blocks.get(kind, []):
                inequality_pieces.append(
                    rows[EXPONENTIAL_TO_ECOS] if kind == "Exponential" else rows)
        inequality_rows = concatenate_rows(inequality_pieces)

        E = problem["E"].tocsr()
        offset = numpy.asarray(problem["b"], float)
        dimensions = {
            "l": int(sum(size for size, _p, _r in blocks.get("Nonneg", []))),
            "q": [int(size) for size, _p, _r in blocks.get("SecondOrder", [])],
            "e": len(blocks.get("Exponential", [])),
        }
        arguments = dict(c=numpy.asarray(problem["q"], float),
                         G=E[inequality_rows].tocsc(), h=offset[inequality_rows],
                         dims=dimensions, verbose=False,
                         **imposed(feastol=eps_abs, abstol=eps_abs,
                                   reltol=SMALLEST_RELATIVE_TOLERANCE,
                                   max_iters=UNBOUNDED_ITERATIONS))
        if equality_rows.size:
            arguments["A"] = E[equality_rows].tocsc()
            arguments["b"] = offset[equality_rows]
        solution = backend.solve(**arguments)

        status = STATUS_NAMES.get(int(solution["info"]["exitFlag"]), "max_iter")
        dual = numpy.zeros(problem["E"].shape[0])
        if equality_rows.size:
            dual[equality_rows] = numpy.asarray(solution["y"], float)
        if inequality_rows.size:
            dual[inequality_rows] = numpy.asarray(solution["z"], float)
        return {"status": status, "x": numpy.asarray(solution["x"], float), "z": dual}
