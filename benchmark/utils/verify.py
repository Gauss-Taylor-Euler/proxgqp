import numpy

from cones import project_dual

INFEASIBILITY_STATUSES = ("primal_infeasible", "dual_infeasible")


def residuals(problem, x, z):
    P, q, E, f = problem["P"], problem["q"], problem["E"], problem["f"]
    cones = problem["cones"]
    slack = f - E @ x
    infinity_norm = lambda vector: float(numpy.linalg.norm(vector, numpy.inf))
    primal_residual = infinity_norm(project_dual(cones, -slack))
    dual_residual = infinity_norm(P @ x + q + E.T @ z)
    dual_cone_violation = infinity_norm(z - project_dual(cones, z))
    complementarity = abs(float(slack @ z))
    return primal_residual, dual_residual, dual_cone_violation, complementarity


def scales(problem, x, z):
    """What each residual is measured against.  The three residuals scale with
    the terms they are a difference of; the gap is a product and so scales with
    the objective, which is why holding it to the same absolute bar as a
    residual asks it for an order of magnitude more accuracy than the rest.
    The gap carries its own tolerance for the same reason PIQP and Clarabel
    give it one: it is a different quantity, not a fourth residual."""
    P, q, E, f = problem["P"], problem["q"], problem["E"], problem["f"]
    infinity_norm = lambda vector: float(numpy.linalg.norm(vector, numpy.inf))
    constraint_image = E @ x
    slack = f - constraint_image
    quadratic = P @ x
    primal_scale = max(infinity_norm(constraint_image), infinity_norm(slack),
                       infinity_norm(f))
    dual_scale = max(infinity_norm(quadratic), infinity_norm(q),
                     infinity_norm(E.T @ z))
    cone_scale = infinity_norm(z)
    primal_objective = float(0.5 * (x @ quadratic) + q @ x)
    dual_objective = float(-0.5 * (x @ quadratic) - f @ z)
    gap_scale = max(abs(primal_objective), abs(dual_objective))
    return primal_scale, dual_scale, cone_scale, gap_scale


def objective(problem, x):
    P, q = problem["P"], problem["q"]
    constant = problem.get("objective_constant", 0.0)
    return float(0.5 * (x @ (P @ x)) + q @ x + constant)


def empty_verdict():
    return dict(primal_residual=None, dual_residual=None,
                dual_cone_violation=None, complementarity=None,
                objective=None, verified=False)


def verify(problem, result, eps_abs, eps_rel=None, eps_duality_gap_abs=None,
           eps_duality_gap_rel=None):
    if eps_rel is None:
        eps_rel = eps_abs
    if eps_duality_gap_abs is None:
        eps_duality_gap_abs = eps_abs
    if eps_duality_gap_rel is None:
        eps_duality_gap_rel = eps_rel
    status = result.get("status")
    x = result.get("x")
    z = result.get("z")
    if x is None or z is None or status in INFEASIBILITY_STATUSES:
        return empty_verdict()
    x = numpy.asarray(x, float).ravel()
    z = numpy.asarray(z, float).ravel()
    if x.size != problem["P"].shape[0] or z.size != problem["E"].shape[0]:
        return empty_verdict()
    primal_residual, dual_residual, dual_cone_violation, complementarity = residuals(
        problem, x, z)
    primal_scale, dual_scale, cone_scale, gap_scale = scales(problem, x, z)
    passed = (primal_residual <= eps_abs + eps_rel * primal_scale and
              dual_residual <= eps_abs + eps_rel * dual_scale and
              dual_cone_violation <= eps_abs + eps_rel * cone_scale and
              complementarity <= eps_duality_gap_abs
              + eps_duality_gap_rel * gap_scale)
    return dict(primal_residual=primal_residual,
                dual_residual=dual_residual,
                dual_cone_violation=dual_cone_violation,
                complementarity=complementarity,
                objective=objective(problem, x),
                verified=bool(passed))
