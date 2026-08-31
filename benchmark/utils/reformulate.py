import numpy
import scipy.sparse


def cone_kinds(cones):
    return {kind for kind, _size, _params in cones}


def equality_inequality_rows(cones):
    equality_rows = []
    inequality_rows = []
    offset = 0
    for kind, size, _params in cones:
        rows = range(offset, offset + size)
        if kind == "Zero":
            equality_rows.extend(rows)
        elif kind == "Nonneg":
            inequality_rows.extend(rows)
        else:
            raise ValueError("cone %r is not linear" % kind)
        offset += size
    return numpy.array(equality_rows, dtype=numpy.int64), \
        numpy.array(inequality_rows, dtype=numpy.int64)


def split_linear(problem):
    E = problem["E"].tocsr()
    f = numpy.asarray(problem["f"], float)
    equality_rows, inequality_rows = equality_inequality_rows(problem["cones"])
    equality_matrix = E[equality_rows].tocsc()
    inequality_matrix = E[inequality_rows].tocsc()
    return (equality_matrix, f[equality_rows],
            inequality_matrix, f[inequality_rows],
            equality_rows, inequality_rows)


def gather_dual(row_count, equality_rows, equality_dual, inequality_rows, inequality_dual):
    dual = numpy.zeros(row_count)
    if equality_rows.size:
        dual[equality_rows] = numpy.asarray(equality_dual, float).ravel()
    if inequality_rows.size:
        dual[inequality_rows] = numpy.asarray(inequality_dual, float).ravel()
    return dual


def rows_by_kind(cones):
    from cones import block_rows
    blocks = {}
    offset = 0
    for kind, size, params in cones:
        rows = block_rows(kind, size)
        blocks.setdefault(kind, []).append(
            (size, params, numpy.arange(offset, offset + rows, dtype=numpy.int64)))
        offset += rows
    return blocks


def order_rows(cones, kind_order):
    blocks = rows_by_kind(cones)
    unknown = set(blocks) - set(kind_order)
    if unknown:
        raise ValueError("cones %s cannot be ordered" % sorted(unknown))
    pieces = []
    for kind in kind_order:
        for _size, _params, rows in blocks.get(kind, []):
            pieces.append(rows)
    if not pieces:
        return numpy.zeros(0, dtype=numpy.int64), blocks
    return numpy.concatenate(pieces), blocks
