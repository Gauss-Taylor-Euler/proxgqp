#include "globalisation/merit.hpp"

#include "cone/kernel.hpp"

namespace proxgqp {

void Merit::setup(const Cones& given_cones, Index columns, Index rows) {
  cones = &given_cones;
  offsets = block_offsets(given_cones);
  direction_x.setZero(columns);
  direction_slack.setZero(rows);
  trial_x.setZero(columns);
  trial_slack.setZero(rows);
  residual.setZero(rows);
  multiplier.setZero(rows);
  reference_point.setZero(rows);
  projected.setZero(rows);
  gradient_x.setZero(columns);
  gradient_slack.setZero(rows);
  work_rows.setZero(rows);
}

void Merit::bind(const ProblemData& given_data, const Iterate& given_iterate,
                 const ConstVectorRef& dx, const ConstVectorRef& ds) {
  data = &given_data;
  iterate = &given_iterate;
  direction_x = dx;
  direction_slack = ds;

  work_rows.noalias() = *data->E * direction_x;
  work_rows += direction_slack;
  trial_x.noalias() = *data->P * direction_x;
  curvature = direction_x.dot(trial_x) +
              iterate->rho * direction_x.squaredNorm() +
              iterate->proximal_slack * direction_slack.squaredNorm() +
              work_rows.dot(work_rows.cwiseQuotient(iterate->equality_penalty));
}

Scalar Merit::slope(Scalar step) {
  trial_x = iterate->x + step * direction_x;
  trial_slack = iterate->s + step * direction_slack;

  residual.noalias() = *data->E * trial_x;
  residual += trial_slack - *data->f;
  multiplier =
      iterate->y_centre + residual.cwiseQuotient(iterate->equality_penalty);

  reference_point =
      iterate->cone_penalty.cwiseProduct(iterate->z_centre) - trial_slack;
  for (std::size_t j = 0; j < cones->size(); ++j) {
    const Index start = offsets[j];
    const Index length = offsets[j + 1] - offsets[j];
    evaluate_projection((*cones)[j], reference_point.segment(start, length),
                        projected.segment(start, length));
  }

  gradient_x.noalias() = *data->P * trial_x;
  gradient_x += *data->q + iterate->rho * (trial_x - iterate->x_centre);
  gradient_x.noalias() += data->E->transpose() * multiplier;

  gradient_slack =
      iterate->proximal_slack * (trial_slack - iterate->s_centre) + multiplier -
      projected.cwiseQuotient(iterate->cone_penalty);

  return gradient_x.dot(direction_x) + gradient_slack.dot(direction_slack);
}

}
