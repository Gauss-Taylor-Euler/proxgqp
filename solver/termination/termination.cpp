#include "termination/termination.hpp"

#include <algorithm>
#include <cmath>

namespace proxgqp {

const char* status_name(Status status) {
  switch (status) {
    case Status::Solved: return "solved";
    case Status::PrimalInfeasible: return "primal_infeasible";
    case Status::DualInfeasible: return "dual_infeasible";
    case Status::MaxNewton: return "max_newton";
    case Status::NumericalFailure: return "numerical_failure";
    default: return "max_iter_outer";
  }
}

void Termination::setup(const Cones& cones, Index columns, Index rows) {
  offsets = block_offsets(cones);
  recomputed_slack.setZero(rows);
  projected.setZero(rows);
  violation.setZero(rows);
  work_rows.setZero(rows);
  stationarity.setZero(columns);
  work_columns.setZero(columns);
  constraint_image.setZero(columns);
}

KktReport Termination::evaluate(const ProblemData& data, const Cones& cones,
                                const Vector& x, const Vector& z,
                                const TerminationSettings& settings) {
  KktReport report;
  work_rows.noalias() = *data.E * x;
  const Scalar constraint_size = infinity_norm(work_rows);
  recomputed_slack = *data.b - work_rows;

  work_rows = -recomputed_slack;
  evaluate_projection_all(cones, offsets, work_rows, violation);
  violation = violation.cwiseAbs();
  for (std::size_t block = 0; block < cones.size(); ++block) {
    if (!is_curved(cones[block])) continue;
    const Index start = offsets[block];
    const Index length = offsets[block + 1] - offsets[block];
    violation.segment(start, length)
        .setConstant(violation.segment(start, length).lpNorm<Eigen::Infinity>());
  }
  report.primal_residual = infinity_norm(violation);

  work_columns.noalias() = *data.P * x;
  constraint_image.noalias() = data.E->transpose() * z;
  stationarity = work_columns + *data.q + constraint_image;
  report.dual_residual = infinity_norm(stationarity);

  evaluate_projection_all(cones, offsets, z, projected);
  projected = z - projected;
  report.dual_cone_violation = infinity_norm(projected);

  report.complementarity = std::abs(recomputed_slack.dot(z));

  const Scalar quadratic_size = infinity_norm(work_columns);
  const Scalar quadratic_form = x.dot(work_columns);
  report.objective = 0.5 * quadratic_form + data.q->dot(x);

  const Scalar primal_scale =
      std::max({constraint_size, infinity_norm(recomputed_slack),
                infinity_norm(*data.b)});
  const Scalar dual_scale =
      std::max({quadratic_size, infinity_norm(*data.q),
                infinity_norm(constraint_image)});
  const Scalar cone_scale = infinity_norm(z);
  const Scalar dual_objective = -0.5 * quadratic_form - data.b->dot(z);
  const Scalar gap_scale =
      std::max(std::abs(report.objective), std::abs(dual_objective));

  const Scalar absolute = settings.eps_absolute;
  const Scalar relative = settings.eps_relative;
  report.relative_primal_residual =
      report.primal_residual / std::max(primal_scale, Scalar(1));
  report.relative_dual_residual =
      report.dual_residual / std::max(dual_scale, Scalar(1));
  report.converged =
      report.primal_residual <= absolute + relative * primal_scale &&
      report.dual_residual <= absolute + relative * dual_scale &&
      report.dual_cone_violation <= absolute + relative * cone_scale &&
      report.complementarity <= settings.eps_gap_absolute +
                                    settings.eps_gap_relative * gap_scale;
  return report;
}

bool Termination::primal_infeasible(const ProblemData& data, const Cones& cones,
                                    const Vector& dual_step,
                                    const TerminationSettings& settings) {
  const Scalar size = infinity_norm(dual_step);
  if (size <= 0.0) return false;
  work_rows = dual_step / size;
  evaluate_projection_all(cones, offsets, work_rows, projected);
  projected = work_rows - projected;
  if (infinity_norm(projected) > settings.eps_primal_infeasible) return false;
  work_columns.noalias() = data.E->transpose() * work_rows;
  if (infinity_norm(work_columns) > settings.eps_primal_infeasible)
    return false;
  return data.b->dot(work_rows) < -settings.eps_primal_infeasible;
}

bool Termination::dual_infeasible(const ProblemData& data, const Cones& cones,
                                  const Vector& primal_step,
                                  const TerminationSettings& settings) {
  const Scalar size = infinity_norm(primal_step);
  if (size <= 0.0) return false;
  work_columns = primal_step / size;
  stationarity.noalias() = *data.P * work_columns;
  if (infinity_norm(stationarity) > settings.eps_dual_infeasible) return false;
  if (data.q->dot(work_columns) > -settings.eps_dual_infeasible) return false;
  work_rows.noalias() = *data.E * work_columns;
  evaluate_projection_all(cones, offsets, work_rows, projected);
  return infinity_norm(projected) <= settings.eps_dual_infeasible;
}

}
