#include <algorithm>
#include <limits>

#include "cone/kernel.hpp"

namespace proxgqp::detail {

Index rows_impl(const Nonneg& cone) { return static_cast<Index>(cone.dim); }

std::size_t degree_impl(const Nonneg& cone) { return cone.dim; }

bool is_curved_impl(const Nonneg&) { return false; }

bool has_interior_impl(const Nonneg&) { return true; }

bool contains_impl(const Nonneg&, ConeSide, const ConstVectorRef& v) {
  return v.minCoeff() >= 0.0;
}

void evaluate_projection_impl(const Nonneg&, const ConstVectorRef& v,
                              VectorRef out) {
  out = v.cwiseMax(0.0);
}

void build_projection_scaling_impl(const Nonneg&, const ConstVectorRef& sigma,
                                   const ConstVectorRef& mu,
                                   BlockScaling& scaling) {
  scaling.source = ScalingSource::Projection;
  scaling.linearisation_point = sigma;
  scaling.penalty = mu;
}

void build_barrier_scaling_impl(const Nonneg&, const ConstVectorRef& s,
                                const ConstVectorRef& z,
                                BlockScaling& scaling) {
  scaling.source = ScalingSource::Barrier;
  scaling.scaling_point = (s.array() / z.array()).sqrt();
  scaling.lambda = (s.array() * z.array()).sqrt();
}

void materialise_operator_impl(const Nonneg&, const BlockScaling& scaling,
                               Scalar rho_p, GBlock& block) {
  block.kind = GBlock::Kind::Diagonal;
  if (scaling.source == ScalingSource::Projection) {
    block.diagonal =
        rho_p + (scaling.linearisation_point.array() > 0.0).cast<Scalar>() /
                    scaling.penalty.array();
    return;
  }
  block.diagonal = rho_p + scaling.scaling_point.array().square().inverse();
}

GBlock::Kind widest_operator_kind_impl(const Nonneg&) {
  return GBlock::Kind::Diagonal;
}

SchurAssembly schur_assembly_impl(const Nonneg&) {
  return SchurAssembly::Materialise;
}

void apply_inverse_operator_impl(const Nonneg&, const BlockScaling&, Scalar,
                                 const ConstVectorRef&, VectorRef) {
  schur_weight_is_materialised();
}

void apply_schur_weight_impl(const Nonneg&, const BlockScaling&, Scalar,
                             const ConstVectorRef&, const ConstVectorRef&,
                             VectorRef) {
  schur_weight_is_materialised();
}

void apply_residual_scaling_impl(const Nonneg&, const BlockScaling& scaling,
                                 const ConstVectorRef& v, VectorRef out) {
  if (scaling.source == ScalingSource::Projection) {
    out = (scaling.linearisation_point.array() > 0.0).select(v, 0.0);
    return;
  }
  out = v.array() /
        (scaling.scaling_point.array() * scaling.lambda.array());
}

void complementarity_residual_impl(const Nonneg&, const BlockScaling& scaling,
                                   Scalar target, VectorRef out) {
  out = scaling.lambda.array().square() - target;
}

void second_order_correction_impl(const Nonneg&, const BlockScaling&,
                                  const ConstVectorRef& ds,
                                  const ConstVectorRef& dz, VectorRef out) {
  out = ds.array() * dz.array();
}

Scalar largest_feasible_step_impl(const Nonneg&, ConeSide,
                                  const ConstVectorRef& v,
                                  const ConstVectorRef& dv) {
  Scalar step = std::numeric_limits<Scalar>::infinity();
  for (Index row = 0; row < v.size(); ++row)
    if (dv(row) < 0.0) step = std::min(step, -v(row) / dv(row));
  return step;
}

void interior_direction_impl(const Nonneg&, ConeSide, VectorRef out) {
  out.setOnes();
}

Scalar margin_impl(const Nonneg&, ConeSide, const ConstVectorRef& v) {
  return v.minCoeff();
}

}
