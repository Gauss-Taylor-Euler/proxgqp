import numpy
import scipy.sparse

from const import SMALLEST_RELATIVE_TOLERANCE, UNBOUNDED_ITERATIONS
from reformulate import order_rows

KIND_ORDER = ("Zero", "Nonneg", "SecondOrder", "PSDTriangle", "Exponential", "Power")

STATUS_NAMES = {
    "solved": "solved",
    "solved (inaccurate - reached max_iters)": "solved",
    "primal infeasible": "primal_infeasible",
    "primal infeasible (inaccurate)": "primal_infeasible",
    "dual infeasible": "dual_infeasible",
    "dual infeasible (inaccurate)": "dual_infeasible",
}


class Scs:

    name = "scs"
    quadratic = True
    cones = KIND_ORDER

    @staticmethod
    def solve(problem, eps_abs, warm_start=None):
        import scs as backend

        row_order, blocks = order_rows(problem["cones"], KIND_ORDER)
        E = problem["E"].tocsr()[row_order].tocsc()
        f = numpy.asarray(problem["f"], float)[row_order]

        cone_sizes = {
            "z": int(sum(size for size, _p, _r in blocks.get("Zero", []))),
            "l": int(sum(size for size, _p, _r in blocks.get("Nonneg", []))),
            "q": [int(size) for size, _p, _r in blocks.get("SecondOrder", [])],
            "s": [int(size) for size, _p, _r in blocks.get("PSDTriangle", [])],
            "ep": len(blocks.get("Exponential", [])),
            "p": [float(params["alpha"]) for _s, params, _r in blocks.get("Power", [])],
        }
        data = {"P": scipy.sparse.triu(problem["P"], format="csc"),
                "A": E, "b": f, "c": numpy.asarray(problem["q"], float)}

        options = dict(eps_abs=eps_abs, eps_rel=SMALLEST_RELATIVE_TOLERANCE,
                       eps_infeas=eps_abs, max_iters=UNBOUNDED_ITERATIONS,
                       verbose=False)
        if warm_start is not None:
            primal = numpy.asarray(warm_start["x"], float)
            dual = numpy.asarray(warm_start["z"], float)[row_order]
            data["x"] = primal
            data["y"] = dual
            data["s"] = f - E @ primal
        solution = backend.solve(data, cone_sizes, **options)
        status = STATUS_NAMES.get(str(solution["info"]["status"]).strip().lower(), "max_iter")
        if solution["x"] is None or solution["y"] is None:
            return {"status": status, "x": None, "z": None}
        dual = numpy.empty(problem["E"].shape[0])
        dual[row_order] = numpy.asarray(solution["y"], float)
        return {"status": status, "x": numpy.asarray(solution["x"], float), "z": dual,
                "inner": solution["info"].get("iter")}
