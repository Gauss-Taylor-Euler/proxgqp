import numpy
import scipy.sparse

from .cones import Cone, cone_rows


def _vector(value, length, name):
    if value is None:
        return numpy.zeros(length, dtype=numpy.float64)
    array = numpy.ascontiguousarray(value, dtype=numpy.float64).ravel()
    if array.size != length:
        raise ValueError("%s has %d entries, expected %d" % (name, array.size, length))
    return array


def _matrix(value, shape, name):
    if value is None:
        matrix = scipy.sparse.csc_matrix(shape, dtype=numpy.float64)
    else:
        matrix = scipy.sparse.csc_matrix(value, dtype=numpy.float64)
    if matrix.shape != shape:
        raise ValueError("%s has shape %s, expected %s" % (name, matrix.shape, shape))
    matrix.sort_indices()
    return matrix


class Problem:

    def __init__(self, P, q, E, f, cones):
        cones = list(cones)
        for cone in cones:
            if not isinstance(cone, Cone):
                raise TypeError("cones must be Cone instances, got %r" % (cone,))
        self.cones = cones

        q = numpy.ascontiguousarray(q, dtype=numpy.float64).ravel()
        self.columns = q.size
        self.rows = cone_rows(cones)

        self.q = q
        self.P = _matrix(P, (self.columns, self.columns), "P")
        self.E = _matrix(E, (self.rows, self.columns), "E")
        self.f = _vector(f, self.rows, "f")

        self.kinds = numpy.array([cone.kind for cone in cones], dtype=numpy.int32)
        self.sizes = numpy.array([cone.size for cone in cones], dtype=numpy.int64)
        self.exponents = numpy.array([cone.exponent for cone in cones],
                                     dtype=numpy.float64)

    def __repr__(self):
        return ("Problem(columns=%d, rows=%d, cones=%d, nnzP=%d, nnzE=%d)"
                % (self.columns, self.rows, len(self.cones), self.P.nnz, self.E.nnz))
