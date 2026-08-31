import math

import numpy
import scipy.sparse

from cones import matrix_to_svec, triangle_side

ACTIVITIES = ("active", "inactive", "degenerate")


def zero_pair(size, rng, activity):
    return numpy.zeros(size), rng.normal(size=size)


def nonneg_pair(size, rng, activity):
    slack = numpy.zeros(size)
    dual = numpy.zeros(size)
    if activity == "degenerate":
        return slack, dual
    choice = rng.random(size)
    if activity == "inactive":
        slack = rng.random(size) + 0.1
        return slack, dual
    inactive = choice > 0.5
    slack[inactive] = rng.random(int(inactive.sum())) + 0.1
    dual[~inactive] = rng.random(int((~inactive).sum())) + 0.1
    return slack, dual


def second_order_pair(size, rng, activity):
    tail = rng.normal(size=size - 1)
    tail_norm = float(numpy.linalg.norm(tail))
    if tail_norm < 1e-12:
        tail = numpy.ones(size - 1)
        tail_norm = float(numpy.linalg.norm(tail))
    if activity == "degenerate":
        return numpy.zeros(size), numpy.zeros(size)
    if activity == "inactive":
        slack = numpy.concatenate([[tail_norm * (1.0 + rng.random())], tail])
        return slack, numpy.zeros(size)
    slack = numpy.concatenate([[tail_norm], tail])
    scale = rng.random() + 0.1
    dual = scale * numpy.concatenate([[tail_norm], -tail]) / tail_norm
    return slack, dual


def psd_pair(side, rng, activity):
    basis, _ = numpy.linalg.qr(rng.normal(size=(side, side)))
    if activity == "degenerate":
        return numpy.zeros(side * (side + 1) // 2), numpy.zeros(side * (side + 1) // 2)
    if activity == "inactive":
        eigenvalues = rng.random(side) + 0.1
        slack = (basis * eigenvalues) @ basis.T
        return matrix_to_svec(slack), numpy.zeros(side * (side + 1) // 2)
    rank = max(1, side // 2)
    slack_eigenvalues = numpy.zeros(side)
    dual_eigenvalues = numpy.zeros(side)
    slack_eigenvalues[:rank] = rng.random(rank) + 0.1
    dual_eigenvalues[rank:] = rng.random(side - rank) + 0.1
    slack = (basis * slack_eigenvalues) @ basis.T
    dual = (basis * dual_eigenvalues) @ basis.T
    return matrix_to_svec(slack), matrix_to_svec(dual)


def exponential_pair(_size, rng, activity):
    if activity == "degenerate":
        return numpy.zeros(3), numpy.zeros(3)
    height = rng.random() + 0.5
    ratio = rng.normal() * 0.5
    growth = math.exp(ratio)
    boundary = numpy.array([ratio * height, height, height * growth])
    if activity == "inactive":
        interior = boundary + numpy.array([0.0, 0.0, rng.random() + 0.1])
        return interior, numpy.zeros(3)
    gradient = numpy.array([growth, growth * (1.0 - ratio), -1.0])
    return boundary, (rng.random() + 0.1) * (-gradient)


def power_pair(_size, rng, activity, exponent):
    if activity == "degenerate":
        return numpy.zeros(3), numpy.zeros(3)
    first = rng.random() + 0.5
    second = rng.random() + 0.5
    magnitude = math.exp(exponent * math.log(first)
                         + (1.0 - exponent) * math.log(second))
    sign = 1.0 if rng.random() > 0.5 else -1.0
    boundary = numpy.array([first, second, sign * magnitude])
    if activity == "inactive":
        interior = numpy.array([first * 1.5, second * 1.5, sign * magnitude])
        return interior, numpy.zeros(3)
    gradient = numpy.array([exponent * magnitude / first,
                            (1.0 - exponent) * magnitude / second,
                            -sign])
    return boundary, (rng.random() + 0.1) * gradient


def complementary_pair(kind, size, params, rng, activity):
    if kind == "Zero":
        return zero_pair(size, rng, activity)
    if kind == "Nonneg":
        return nonneg_pair(size, rng, activity)
    if kind == "SecondOrder":
        return second_order_pair(size, rng, activity)
    if kind == "PSDTriangle":
        return psd_pair(size, rng, activity)
    if kind == "Exponential":
        return exponential_pair(size, rng, activity)
    if kind == "Power":
        return power_pair(size, rng, activity, params["alpha"])
    raise ValueError("unknown cone kind %r" % kind)


def random_psd_matrix(dimension, rank, condition_number, rng):
    if rank <= 0:
        return scipy.sparse.csc_matrix((dimension, dimension))
    basis, _ = numpy.linalg.qr(rng.normal(size=(dimension, dimension)))
    eigenvalues = numpy.zeros(dimension)
    if rank == 1:
        eigenvalues[0] = 1.0
    else:
        eigenvalues[:rank] = numpy.logspace(0.0, -math.log10(condition_number), rank)
    dense = (basis * eigenvalues) @ basis.T
    dense = 0.5 * (dense + dense.T)
    return scipy.sparse.csc_matrix(dense)


def random_sparse_matrix(row_count, column_count, per_row, rng):
    per_row = max(1, min(per_row, column_count))
    rows = numpy.repeat(numpy.arange(row_count), per_row)
    columns = numpy.concatenate(
        [rng.choice(column_count, size=per_row, replace=False) for _ in range(row_count)])
    values = rng.normal(size=rows.size)
    return scipy.sparse.csc_matrix((values, (rows, columns)),
                                   shape=(row_count, column_count))


def build(variable_count, cone_specification, rng, per_row=10,
          objective_rank=None, condition_number=1e3, activity_weights=(0.6, 0.3, 0.1)):
    slacks = []
    duals = []
    cones = []
    for kind, size, params in cone_specification:
        activity = rng.choice(ACTIVITIES, p=activity_weights)
        slack, dual = complementary_pair(kind, size, params, rng, activity)
        slacks.append(slack)
        duals.append(dual)
        cones.append((kind, size, params))
    slack = numpy.concatenate(slacks)
    dual = numpy.concatenate(duals)
    row_count = slack.size

    E = random_sparse_matrix(row_count, variable_count, per_row, rng)
    solution = rng.normal(size=variable_count)
    f = E @ solution + slack
    rank = variable_count if objective_rank is None else objective_rank
    P = random_psd_matrix(variable_count, rank, condition_number, rng)
    q = -(P @ solution) - E.T @ dual

    problem = {"P": P, "q": q, "E": E.tocsc(), "f": f, "cones": cones}
    problem["objective_reference"] = float(
        0.5 * (solution @ (P @ solution)) + q @ solution)
    return problem, solution, dual
