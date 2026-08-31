import math

import numpy
import scipy.sparse

FORMULATIONS = ("mean_variance", "min_deviation", "mean_cvar",
                "mean_evar", "risk_budgeting", "max_diversification")


def covariance(returns):
    centred = returns - returns.mean(axis=0)
    return (centred.T @ centred) / max(1, returns.shape[0] - 1)


def covariance_factor(matrix):
    eigenvalues, eigenvectors = numpy.linalg.eigh(matrix)
    kept = numpy.maximum(eigenvalues, 0.0)
    return eigenvectors * numpy.sqrt(kept)


def stack(blocks, values, cones):
    return {"E": scipy.sparse.vstack(blocks).tocsc(),
            "b": numpy.concatenate(values), "cones": cones}


def budget_and_long_only(asset_count, variable_count):
    budget = scipy.sparse.csc_matrix(
        (numpy.ones(asset_count), (numpy.zeros(asset_count), numpy.arange(asset_count))),
        shape=(1, variable_count))
    long_only = scipy.sparse.hstack([
        -scipy.sparse.identity(asset_count, format="csc"),
        scipy.sparse.csc_matrix((asset_count, variable_count - asset_count))]).tocsc()
    return ([budget, long_only],
            [numpy.ones(1), numpy.zeros(asset_count)],
            [("Zero", 1, {}), ("Nonneg", asset_count, {})])


def mean_variance(returns, risk_aversion=1.0):
    asset_count = returns.shape[1]
    blocks, values, cones = budget_and_long_only(asset_count, asset_count)
    return dict(P=scipy.sparse.csc_matrix(covariance(returns)),
                q=-risk_aversion * returns.mean(axis=0), **stack(blocks, values, cones))


def min_deviation(returns):
    asset_count = returns.shape[1]
    variable_count = asset_count + 1
    factor = covariance_factor(covariance(returns))
    blocks, values, cones = budget_and_long_only(asset_count, variable_count)
    apex = scipy.sparse.csc_matrix(
        (numpy.array([-1.0]), (numpy.zeros(1), numpy.array([asset_count]))),
        shape=(1, variable_count))
    tail = scipy.sparse.hstack([
        scipy.sparse.csc_matrix(-factor.T),
        scipy.sparse.csc_matrix((asset_count, 1))]).tocsc()
    blocks += [apex, tail]
    values += [numpy.zeros(1), numpy.zeros(asset_count)]
    cones += [("SecondOrder", asset_count + 1, {})]
    objective = numpy.zeros(variable_count)
    objective[asset_count] = 1.0
    return dict(P=scipy.sparse.csc_matrix((variable_count, variable_count)),
                q=objective, **stack(blocks, values, cones))


def mean_cvar(returns, beta=0.95, risk_aversion=1.0):
    observation_count, asset_count = returns.shape
    variable_count = asset_count + 1 + observation_count
    blocks, values, cones = budget_and_long_only(asset_count, variable_count)
    losses = scipy.sparse.hstack([
        scipy.sparse.csc_matrix(-returns),
        -scipy.sparse.csc_matrix(numpy.ones((observation_count, 1))),
        -scipy.sparse.identity(observation_count, format="csc")]).tocsc()
    excess = scipy.sparse.hstack([
        scipy.sparse.csc_matrix((observation_count, asset_count + 1)),
        -scipy.sparse.identity(observation_count, format="csc")]).tocsc()
    blocks += [losses, excess]
    values += [numpy.zeros(observation_count), numpy.zeros(observation_count)]
    cones += [("Nonneg", observation_count, {}), ("Nonneg", observation_count, {})]
    objective = numpy.zeros(variable_count)
    objective[:asset_count] = -risk_aversion * returns.mean(axis=0)
    objective[asset_count] = 1.0
    objective[asset_count + 1:] = 1.0 / (observation_count * (1.0 - beta))
    return dict(P=scipy.sparse.csc_matrix((variable_count, variable_count)),
                q=objective, **stack(blocks, values, cones))


def mean_evar(returns, beta=0.95, risk_aversion=1.0):
    observation_count, asset_count = returns.shape
    variable_count = asset_count + 2 + observation_count
    scale_index = asset_count
    width_index = asset_count + 1
    blocks, values, cones = budget_and_long_only(asset_count, variable_count)
    budget_row = scipy.sparse.hstack([
        scipy.sparse.csc_matrix((1, asset_count + 2)),
        scipy.sparse.csc_matrix(numpy.ones((1, observation_count)))]).tocsc()
    width_row = scipy.sparse.csc_matrix(
        (numpy.array([-1.0]), (numpy.zeros(1), numpy.array([width_index]))),
        shape=(1, variable_count))
    blocks += [(budget_row + width_row)]
    values += [numpy.zeros(1)]
    cones += [("Nonneg", 1, {})]
    for observation in range(observation_count):
        rows = numpy.zeros((3, variable_count))
        rows[0, :asset_count] = returns[observation]
        rows[0, scale_index] = 1.0
        rows[1, width_index] = -1.0
        rows[2, asset_count + 2 + observation] = -1.0
        blocks.append(scipy.sparse.csc_matrix(rows))
        values.append(numpy.zeros(3))
        cones.append(("Exponential", 3, {}))
    objective = numpy.zeros(variable_count)
    objective[:asset_count] = -risk_aversion * returns.mean(axis=0)
    objective[scale_index] = 1.0
    objective[width_index] = math.log(1.0 / (observation_count * (1.0 - beta)))
    return dict(P=scipy.sparse.csc_matrix((variable_count, variable_count)),
                q=objective, **stack(blocks, values, cones))


def risk_budgeting(returns, budget=None):
    asset_count = returns.shape[1]
    variable_count = 2 * asset_count
    if budget is None:
        budget = numpy.full(asset_count, 1.0 / asset_count)
    blocks = [scipy.sparse.hstack([
        -scipy.sparse.identity(asset_count, format="csc"),
        scipy.sparse.csc_matrix((asset_count, asset_count))]).tocsc()]
    values = [numpy.zeros(asset_count)]
    cones = [("Nonneg", asset_count, {})]
    logarithm_row = numpy.zeros((1, variable_count))
    logarithm_row[0, asset_count:] = -budget
    blocks.append(scipy.sparse.csc_matrix(logarithm_row))
    values.append(numpy.zeros(1))
    cones.append(("Nonneg", 1, {}))
    for asset in range(asset_count):
        rows = numpy.zeros((3, variable_count))
        rows[0, asset_count + asset] = -1.0
        rows[2, asset] = -1.0
        blocks.append(scipy.sparse.csc_matrix(rows))
        values.append(numpy.array([0.0, 1.0, 0.0]))
        cones.append(("Exponential", 3, {}))
    objective = numpy.zeros(variable_count)
    return dict(P=scipy.sparse.block_diag(
                    [covariance(returns),
                     scipy.sparse.csc_matrix((asset_count, asset_count))]).tocsc(),
                q=objective, **stack(blocks, values, cones))


def max_diversification(returns):
    asset_count = returns.shape[1]
    variable_count = asset_count + 1
    matrix = covariance(returns)
    factor = covariance_factor(matrix)
    deviations = numpy.sqrt(numpy.maximum(numpy.diag(matrix), 1e-12))
    normalisation = scipy.sparse.csc_matrix(
        (deviations, (numpy.zeros(asset_count), numpy.arange(asset_count))),
        shape=(1, variable_count))
    long_only = scipy.sparse.hstack([
        -scipy.sparse.identity(asset_count, format="csc"),
        scipy.sparse.csc_matrix((asset_count, 1))]).tocsc()
    apex = scipy.sparse.csc_matrix(
        (numpy.array([-1.0]), (numpy.zeros(1), numpy.array([asset_count]))),
        shape=(1, variable_count))
    tail = scipy.sparse.hstack([
        scipy.sparse.csc_matrix(-factor.T),
        scipy.sparse.csc_matrix((asset_count, 1))]).tocsc()
    objective = numpy.zeros(variable_count)
    objective[asset_count] = 1.0
    return dict(P=scipy.sparse.csc_matrix((variable_count, variable_count)),
                q=objective,
                **stack([normalisation, long_only, apex, tail],
                        [numpy.ones(1), numpy.zeros(asset_count),
                         numpy.zeros(1), numpy.zeros(asset_count)],
                        [("Zero", 1, {}), ("Nonneg", asset_count, {}),
                         ("SecondOrder", asset_count + 1, {})]))


def build(formulation, returns, **options):
    return {"mean_variance": mean_variance, "min_deviation": min_deviation,
            "mean_cvar": mean_cvar, "mean_evar": mean_evar,
            "risk_budgeting": risk_budgeting,
            "max_diversification": max_diversification}[formulation](returns, **options)
