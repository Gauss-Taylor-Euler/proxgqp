#include "globalisation/merit.hpp"

#include "cone/kernel.hpp"

namespace proxgqp {

void Merit::setup(const Cones& given_cones, Index columns, Index rows) {
  cones = &given_cones;
  offsets = block_offsets(given_cones);
  direction_slack.setZero(rows);
  direction_image.setZero(rows);
  residual.setZero(rows);
  multiplier.setZero(rows);
  reference_base.setZero(rows);
  reference_point.setZero(rows);
  projected.setZero(rows);
  slack_over_penalty.setZero(rows);
  objective_image.setZero(columns);
}

void Merit::bind(const ProblemData& data, const Iterate& iterate,
                 const ConstVectorRef& dx, const ConstVectorRef& ds) {
  direction_slack = ds;

  direction_image.noalias() = *data.E * dx;
  direction_image += ds;
  objective_image.noalias() = *data.P * dx;
  curvature = objective_image.dot(dx) + iterate.rho * dx.squaredNorm() +
              iterate.proximal_slack * ds.squaredNorm() +
              direction_image.dot(
                  direction_image.cwiseQuotient(iterate.equality_penalty));

  residual.noalias() = *data.E * iterate.x;
  residual += iterate.s - *data.b;
  multiplier =
      iterate.y_centre + residual.cwiseQuotient(iterate.equality_penalty);

  objective_image.noalias() = *data.P * iterate.x;
  constant = objective_image.dot(dx) + data.q->dot(dx) +
             iterate.rho * (iterate.x - iterate.x_centre).dot(dx) +
             iterate.proximal_slack * (iterate.s - iterate.s_centre).dot(ds) +
             multiplier.dot(direction_image);

  reference_base =
      iterate.cone_penalty.cwiseProduct(iterate.z_centre) - iterate.s;
  slack_over_penalty = ds.cwiseQuotient(iterate.cone_penalty);
  penalty = iterate.cone_penalty;
  evaluate_projection_all(*cones, offsets, reference_base, projected);
  base_energy = projected.dot(projected.cwiseQuotient(penalty));
}

Scalar Merit::decrease(Scalar step) {
  reference_point = reference_base - step * direction_slack;
  evaluate_projection_all(*cones, offsets, reference_point, projected);
  const Scalar moved = projected.dot(projected.cwiseQuotient(penalty));
  return step * constant + 0.5 * step * step * curvature +
         0.5 * (moved - base_energy);
}

Scalar Merit::slope(Scalar step) {
  reference_point = reference_base - step * direction_slack;
  evaluate_projection_all(*cones, offsets, reference_point, projected);
  return constant + step * curvature - projected.dot(slack_over_penalty);
}

}
