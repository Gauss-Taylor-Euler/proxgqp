import numpy

from . import _core
from .problem import Problem
from .settings import make_settings

STATUS = {_core.Status.solved: "solved",
          _core.Status.primal_infeasible: "primal_infeasible",
          _core.Status.dual_infeasible: "dual_infeasible",
          _core.Status.max_iterations: "max_iterations",
          _core.Status.numerical_failure: "numerical_failure"}


class Solution:

    def __init__(self, results):
        self.status = STATUS.get(results.status, "numerical_failure")
        self.x = numpy.asarray(results.x)
        self.s = numpy.asarray(results.s)
        self.z = numpy.asarray(results.z)
        self.objective = results.objective
        self.primal_residual = results.primal_residual
        self.dual_residual = results.dual_residual
        self.dual_cone_violation = results.dual_cone_violation
        self.complementarity = results.complementarity
        self.outer_iterations = results.outer_iterations
        self.inner_iterations = results.inner_iterations
        self.seconds = results.seconds

    @property
    def solved(self):
        return self.status == "solved"

    @property
    def residuals(self):
        return {"primal": self.primal_residual,
                "dual": self.dual_residual,
                "cone": self.dual_cone_violation,
                "complementarity": self.complementarity}

    def __repr__(self):
        return ("Solution(status=%r, objective=%.8g, outer=%d, inner=%d, "
                "seconds=%.4f)" % (self.status, self.objective,
                                   self.outer_iterations,
                                   self.inner_iterations, self.seconds))


def _warm_triple(problem, warm_start):
    if warm_start is None:
        return None
    if isinstance(warm_start, Solution):
        x, s, z = warm_start.x, warm_start.s, warm_start.z
    elif isinstance(warm_start, dict):
        missing = [key for key in ("x", "s", "z") if key not in warm_start]
        if missing:
            raise ValueError("warm_start is missing %s" % (", ".join(missing),))
        x, s, z = warm_start["x"], warm_start["s"], warm_start["z"]
    else:
        x, s, z = warm_start

    triple = tuple(numpy.ascontiguousarray(part, dtype=numpy.float64).ravel()
                   for part in (x, s, z))
    expected = (problem.columns, problem.rows, problem.rows)
    for part, length, name in zip(triple, expected, ("x", "s", "z")):
        if part.size != length:
            raise ValueError("warm_start %s has %d entries, expected %d"
                             % (name, part.size, length))
    return triple


def solve(problem, preset=None, warm_start=None, settings=None, **knobs):
    if not isinstance(problem, Problem):
        raise TypeError("solve expects a Problem, got %r" % (type(problem).__name__,))
    if settings is None:
        settings = make_settings(preset, **knobs)
    elif preset is not None or knobs:
        raise ValueError("pass either settings or preset/knobs, not both")

    results = _core.solve(
        problem.columns, problem.rows,
        problem.P.indptr.astype(numpy.int32),
        problem.P.indices.astype(numpy.int32),
        numpy.ascontiguousarray(problem.P.data, dtype=numpy.float64),
        problem.q,
        problem.E.indptr.astype(numpy.int32),
        problem.E.indices.astype(numpy.int32),
        numpy.ascontiguousarray(problem.E.data, dtype=numpy.float64),
        problem.f,
        problem.kinds, problem.sizes, problem.exponents,
        settings, _warm_triple(problem, warm_start))
    return Solution(results)


def solve_qp(P, q, A=None, b=None, G=None, h=None, preset=None,
             warm_start=None, settings=None, **knobs):
    import scipy.sparse

    from .cones import Nonneg, Zero

    blocks = []
    offsets = []
    cones = []
    columns = numpy.ascontiguousarray(q, dtype=numpy.float64).ravel().size

    if A is not None:
        equality = scipy.sparse.csc_matrix(A, dtype=numpy.float64)
        blocks.append(equality)
        offsets.append(numpy.ascontiguousarray(b, dtype=numpy.float64).ravel())
        cones.append(Zero(equality.shape[0]))
    if G is not None:
        inequality = scipy.sparse.csc_matrix(G, dtype=numpy.float64)
        blocks.append(inequality)
        offsets.append(numpy.ascontiguousarray(h, dtype=numpy.float64).ravel())
        cones.append(Nonneg(inequality.shape[0]))

    if blocks:
        E = scipy.sparse.vstack(blocks, format="csc")
        f = numpy.concatenate(offsets)
    else:
        E = scipy.sparse.csc_matrix((0, columns), dtype=numpy.float64)
        f = numpy.zeros(0, dtype=numpy.float64)

    problem = Problem(P, q, E, f, cones)
    return solve(problem, preset=preset, warm_start=warm_start,
                 settings=settings, **knobs)
