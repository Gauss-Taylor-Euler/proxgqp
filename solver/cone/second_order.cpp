#include <algorithm>
#include <cmath>
#include <limits>

#include "cone/kernel.hpp"

namespace proxgqp::detail {
namespace {

constexpr Scalar kTiny = std::numeric_limits<Scalar>::min();

Scalar lorentz_determinant(const ConstVectorRef& v) {
  return v(0) * v(0) - v.tail(v.size() - 1).squaredNorm();
}

void apply_scaling_point(const ConstVectorRef& scaling_point,
                         const ConstVectorRef& vector, VectorRef out) {
  const Index dim = vector.size();
  const auto scaling_tail = scaling_point.tail(dim - 1);
  const Scalar scaling_apex = scaling_point(0);
  const Scalar apex = vector(0);
  const Scalar tail_inner_product = scaling_tail.dot(vector.tail(dim - 1));
  out(0) = scaling_apex * apex + tail_inner_product;
  out.tail(dim - 1) =
      apex * scaling_tail + vector.tail(dim - 1) +
      scaling_tail * (tail_inner_product / (1.0 + scaling_apex));
}

void solve_arrow_system(const ConstVectorRef& lambda,
                        const ConstVectorRef& right_hand_side,
                        VectorRef solution) {
  const Index dim = right_hand_side.size();
  const auto lambda_tail = lambda.tail(dim - 1);
  const Scalar lambda_apex = lambda(0);
  const Scalar apex = right_hand_side(0);
  const Scalar determinant =
      lambda_apex * lambda_apex - lambda_tail.squaredNorm();
  const Scalar tail_inner_product =
      lambda_tail.dot(right_hand_side.tail(dim - 1));
  solution(0) = (lambda_apex * apex - tail_inner_product) / determinant;
  solution.tail(dim - 1) =
      (-apex * lambda_tail + (determinant * right_hand_side.tail(dim - 1) +
                              lambda_tail * tail_inner_product) /
                                 lambda_apex) /
      determinant;
}

void jordan_product(const ConstVectorRef& left, const ConstVectorRef& right,
                    VectorRef out) {
  const Index dim = left.size();
  const Scalar inner_product = left.dot(right);
  out.tail(dim - 1) =
      left(0) * right.tail(dim - 1) + right(0) * left.tail(dim - 1);
  out(0) = inner_product;
}

void apply_projection_jacobian(const ConstVectorRef& linearisation_point,
                               const ConstVectorRef& vector, VectorRef out) {
  const Index dim = linearisation_point.size();
  const Scalar apex = linearisation_point(0);
  const Scalar tail_norm = linearisation_point.tail(dim - 1).norm();
  if (tail_norm <= kTiny) {
    if (apex >= 0.0) out = vector; else out.setZero();
    return;
  }
  if (tail_norm < apex) { out = vector; return; }
  if (tail_norm < -apex) { out.setZero(); return; }

  const Scalar projection_coefficient = 0.5 * (1.0 + apex / tail_norm);
  const auto unit_tail = linearisation_point.tail(dim - 1) / tail_norm;
  const Scalar tail_projection = unit_tail.dot(vector.tail(dim - 1));
  const Scalar tail_coefficient =
      0.5 * (vector(0) - (apex / tail_norm) * tail_projection) / tail_norm;
  out(0) = 0.5 * tail_projection + 0.5 * vector(0);
  out.tail(dim - 1).noalias() =
      projection_coefficient * vector.tail(dim - 1) +
      tail_coefficient * tail_norm * unit_tail;
}

}

Index rows_impl(const SecondOrder& cone) {
  return static_cast<Index>(cone.dim);
}

std::size_t degree_impl(const SecondOrder&) { return 2; }

bool is_curved_impl(const SecondOrder&) { return true; }

bool has_interior_impl(const SecondOrder&) { return true; }

bool contains_impl(const SecondOrder&, ConeSide, const ConstVectorRef& v) {
  return v.tail(v.size() - 1).norm() <= v(0);
}

void evaluate_projection_impl(const SecondOrder& cone, const ConstVectorRef& v,
                              VectorRef out) {
  const Index dim = static_cast<Index>(cone.dim);
  const Scalar apex = v(0);
  const Scalar tail_norm = v.tail(dim - 1).norm();
  if (tail_norm <= apex) { out = v; return; }
  if (tail_norm <= -apex) { out.setZero(); return; }
  const Scalar projection_coefficient = 0.5 * (1.0 + apex / tail_norm);
  out(0) = projection_coefficient * tail_norm;
  out.tail(dim - 1) = projection_coefficient * v.tail(dim - 1);
}

void build_projection_scaling_impl(const SecondOrder&,
                                   const ConstVectorRef& sigma,
                                   const ConstVectorRef& mu,
                                   BlockScaling& scaling) {
  scaling.source = ScalingSource::Projection;
  scaling.linearisation_point = sigma;
  scaling.penalty = mu;
}

void build_barrier_scaling_impl(const SecondOrder&, const ConstVectorRef& s,
                                const ConstVectorRef& z,
                                BlockScaling& scaling) {
  const Index dim = s.size();
  const Scalar root_determinant_s = std::sqrt(lorentz_determinant(s));
  const Scalar root_determinant_z = std::sqrt(lorentz_determinant(z));
  const Vector unit_s = s / root_determinant_s;
  const Vector unit_z = z / root_determinant_z;
  const Scalar gamma = std::sqrt((1.0 + unit_s.dot(unit_z)) / 2.0);

  scaling.source = ScalingSource::Barrier;
  scaling.scaling_point.resize(dim);
  scaling.scaling_point(0) = (unit_z(0) + unit_s(0)) / (2.0 * gamma);
  scaling.scaling_point.tail(dim - 1) =
      (unit_z.tail(dim - 1) - unit_s.tail(dim - 1)) / (2.0 * gamma);
  scaling.scaling_factor = std::sqrt(root_determinant_z / root_determinant_s);

  scaling.lambda.resize(dim);
  apply_scaling_point(scaling.scaling_point, s, scaling.lambda);
  scaling.lambda *= scaling.scaling_factor;
}

void materialise_operator_impl(const SecondOrder& cone,
                               const BlockScaling& scaling, Scalar rho_p,
                               GBlock& block) {
  const Index dim = static_cast<Index>(cone.dim);
  if (scaling.source == ScalingSource::Barrier) {
    const Scalar factor_squared =
        scaling.scaling_factor * scaling.scaling_factor;
    block.kind = GBlock::Kind::DiagPlusLowRank;
    block.delta = rho_p + factor_squared;
    block.low_rank_columns.resize(dim, 2);
    block.low_rank_columns.col(0) = scaling.scaling_point;
    block.low_rank_columns.col(1).setZero();
    block.low_rank_columns(0, 1) = 1.0;
    block.low_rank_middle.resize(2, 2);
    block.low_rank_middle << 2.0 * factor_squared, 0.0, 0.0,
        -2.0 * factor_squared;
    return;
  }

  const Scalar penalty = scaling.penalty(0);
  const auto& sigma = scaling.linearisation_point;
  const Scalar apex = sigma(0);
  const Scalar tail_norm = sigma.tail(dim - 1).norm();

  const auto constant_diagonal = [&](Scalar value) {
    block.kind = GBlock::Kind::Diagonal;
    block.diagonal = Vector::Constant(dim, value);
  };
  if (tail_norm <= kTiny) {
    constant_diagonal(rho_p + (apex >= 0.0 ? 1.0 / penalty : 0.0));
    return;
  }
  if (tail_norm < apex) { constant_diagonal(rho_p + 1.0 / penalty); return; }
  if (tail_norm < -apex) { constant_diagonal(rho_p); return; }

  const Scalar projection_coefficient = 0.5 * (1.0 + apex / tail_norm);
  block.kind = GBlock::Kind::DiagPlusLowRank;
  block.delta = rho_p + projection_coefficient / penalty;
  block.low_rank_columns.setZero(dim, 2);
  block.low_rank_columns.col(0).tail(dim - 1) =
      sigma.tail(dim - 1) / tail_norm;
  block.low_rank_columns(0, 1) = 1.0;
  block.low_rank_middle.resize(2, 2);
  block.low_rank_middle << -0.5 * apex / tail_norm, 0.5, 0.5,
      0.5 - projection_coefficient;
  block.low_rank_middle /= penalty;
}

GBlock::Kind widest_operator_kind_impl(const SecondOrder&) {
  return GBlock::Kind::DiagPlusLowRank;
}

SchurAssembly schur_assembly_impl(const SecondOrder&) {
  return SchurAssembly::Materialise;
}


void apply_schur_weight_impl(const SecondOrder&, const BlockScaling&, Scalar,
                             const ConstVectorRef&, const ConstVectorRef&,
                             VectorRef) {
  schur_weight_is_materialised();
}

void apply_residual_scaling_impl(const SecondOrder&,
                                 const BlockScaling& scaling,
                                 const ConstVectorRef& v, VectorRef out) {
  if (scaling.source == ScalingSource::Projection) {
    apply_projection_jacobian(scaling.linearisation_point, v, out);
    return;
  }
  Vector arrow_solution(v.size());
  solve_arrow_system(scaling.lambda, v, arrow_solution);
  apply_scaling_point(scaling.scaling_point, arrow_solution, out);
  out *= scaling.scaling_factor;
}

void complementarity_residual_impl(const SecondOrder&,
                                   const BlockScaling& scaling, Scalar target,
                                   VectorRef out) {
  jordan_product(scaling.lambda, scaling.lambda, out);
  out(0) -= 2.0 * target;
}

void second_order_correction_impl(const SecondOrder&,
                                  const BlockScaling& scaling,
                                  const ConstVectorRef& ds,
                                  const ConstVectorRef& dz, VectorRef out) {
  const Index dim = ds.size();
  Vector reflected_point = scaling.scaling_point;
  reflected_point.tail(dim - 1) *= -1.0;
  Vector scaled_ds(dim);
  Vector scaled_dz(dim);
  apply_scaling_point(scaling.scaling_point, ds, scaled_ds);
  scaled_ds *= scaling.scaling_factor;
  apply_scaling_point(reflected_point, dz, scaled_dz);
  scaled_dz /= scaling.scaling_factor;
  jordan_product(scaled_ds, scaled_dz, out);
}

Scalar largest_feasible_step_impl(const SecondOrder&, ConeSide,
                                  const ConstVectorRef& v,
                                  const ConstVectorRef& dv) {
  const Scalar unbounded = std::numeric_limits<Scalar>::infinity();
  const Index dim = v.size();
  const Scalar quadratic = lorentz_determinant(dv);
  const Scalar linear =
      2.0 * (v(0) * dv(0) - v.tail(dim - 1).dot(dv.tail(dim - 1)));
  const Scalar constant = lorentz_determinant(v);
  Scalar limit = unbounded;
  if (std::abs(quadratic) <= kTiny) {
    if (linear < 0.0) limit = -constant / linear;
  } else {
    const Scalar discriminant = linear * linear - 4.0 * quadratic * constant;
    if (discriminant >= 0.0) {
      const Scalar discriminant_root = std::sqrt(discriminant);
      for (Scalar candidate :
           {(-linear + discriminant_root) / (2.0 * quadratic),
            (-linear - discriminant_root) / (2.0 * quadratic)})
        if (candidate > 0.0) limit = std::min(limit, candidate);
    }
  }
  if (dv(0) < 0.0) limit = std::min(limit, -v(0) / dv(0));
  return limit;
}

void interior_direction_impl(const SecondOrder&, ConeSide, VectorRef out) {
  out.setZero();
  out(0) = 1.0;
}

Scalar margin_impl(const SecondOrder&, ConeSide, const ConstVectorRef& v) {
  return v(0) - v.tail(v.size() - 1).norm();
}

}
