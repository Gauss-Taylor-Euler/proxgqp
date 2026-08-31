#include "termination/termination.hpp"

#include <algorithm>
#include <cmath>

namespace proxgqp {

const char* status_name(Status status) {
  switch (status) {
    case Status::Solved: return "solved";
    case Status::PrimalInfeasible: return "primal_infeasible";
    case Status::DualInfeasible: return "dual_infeasible";
    case Status::NumericalFailure: return "numerical_failure";
    default: return "max_iterations";
  }
}

void Termination::setup(const Cones& cones, Index columns, Index rows) {
  offsets = block_offsets(cones);
  recomputed_slack.setZero(rows);
  projected.setZero(rows);
  work_rows.setZero(rows);
  stationarity.setZero(columns);
  work_columns.setZero(columns);
}

KktReport Termination::evaluate(const ProblemData& data, const Cones& cones,
                                const Vector& x, const Vector& z,
                                const TerminationSettings& settings) {
  KktReport report;
  work_rows.noalias() = *data.E * x;
  const Scalar constraint_size = infinity_norm(work_rows);
  recomputed_slack = *data.f - work_rows;

  work_rows = -recomputed_slack;
  evaluate_projection_all(cones, offsets, work_rows, projected);
  report.primal_residual = infinity_norm(projected);

  stationarity.noalias() = *data.P * x;
  stationarity += *data.q;
  stationarity.noalias() += data.E->transpose() * z;
  report.dual_residual = infinity_norm(stationarity);

  evaluate_projection_all(cones, offsets, z, projected);
  report.dual_cone_violation = infinity_norm(Vector(z - projected));

  report.complementarity = std::abs(recomputed_slack.dot(z));

  work_columns.noalias() = *data.P * x;
  const Scalar quadratic_size = infinity_norm(work_columns);
  const Scalar quadratic_form = x.dot(work_columns);
  report.objective = 0.5 * quadratic_form + data.q->dot(x);

  const Scalar primal_scale =
      std::max({constraint_size, infinity_norm(recomputed_slack),
                infinity_norm(*data.f)});
  work_columns.noalias() = data.E->transpose() * z;
  const Scalar dual_scale =
      std::max({quadratic_size, infinity_norm(*data.q),
                infinity_norm(work_columns)});
  const Scalar cone_scale = infinity_norm(z);
  const Scalar dual_objective = -0.5 * quadratic_form - data.f->dot(z);
  const Scalar gap_scale =
      std::max(std::abs(report.objective), std::abs(dual_objective));

  const Scalar absolute = settings.eps_absolute;
  const Scalar relative = settings.eps_relative;
  report.converged =
      report.primal_residual <= absolute + relative * primal_scale &&
      report.dual_residual <= absolute + relative * dual_scale &&
      report.dual_cone_violation <= absolute + relative * cone_scale &&
      report.complementarity <= absolute + relative * gap_scale;
  return report;
}

bool Termination::primal_infeasible(const ProblemData& data, const Cones& cones,
                                    const Vector& dual_step,
                                    const TerminationSettings& settings) {
  const Scalar size = infinity_norm(dual_step);
  if (size <= 0.0) return false;
  work_rows = dual_step / size;
  evaluate_projection_all(cones, offsets, work_rows, projected);
  if (infinity_norm(Vector(work_rows - projected)) >
      settings.eps_primal_infeasible)
    return false;
  work_columns.noalias() = data.E->transpose() * work_rows;
  if (infinity_norm(work_columns) > settings.eps_primal_infeasible)
    return false;
  return data.f->dot(work_rows) < -settings.eps_primal_infeasible;
}

bool Termination::dual_infeasible(const ProblemData& data, const Cones& cones,
                                  const Vector& primal_step,
                                  const TerminationSettings& settings) {
  const Scalar size = infinity_norm(primal_step);
  if (size <= 0.0) return false;
  work_columns = primal_step / size;
  const Vector image = *data.P * work_columns;
  if (image.lpNorm<Eigen::Infinity>() > settings.eps_dual_infeasible)
    return false;
  if (data.q->dot(work_columns) > -settings.eps_dual_infeasible) return false;
  work_rows.noalias() = *data.E * work_columns;
  evaluate_projection_all(cones, offsets, work_rows, projected);
  return infinity_norm(projected) <= settings.eps_dual_infeasible;
}

}
