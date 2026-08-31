#include <algorithm>
#include <cmath>
#include <limits>

#include <Eigen/LU>

#include "cone/curved.hpp"
#include "cone/kernel.hpp"

namespace proxgqp::detail {
namespace {

constexpr int kGridPoints = 1200;
constexpr Scalar kGridLower = -60.0;
constexpr Scalar kGridUpper = 60.0;
constexpr Scalar kRootTolerance = 1e-14;
constexpr int kAnchorEvery = 128;
constexpr Scalar kMembershipTolerance = 1e-9;
constexpr Scalar kDegenerate = 1e-13;

bool inside_interior(const Exponential&, const Vector3& v) {
  const Scalar x = v(0), y = v(1), z = v(2);
  if (!(y > 0.0 && z > 0.0)) return false;
  return y * std::log(z / y) - x > 0.0;
}

bool inside_dual_interior(const Exponential&, const Vector3& v) {
  const Scalar u = v(0), w = v(1), t = v(2);
  if (!(u < 0.0 && t > 0.0)) return false;
  return std::log(-u) + w / u < std::log(t) + 1.0;
}

bool inside_closed(const Vector3& v, Scalar tolerance = kMembershipTolerance) {
  const Scalar x = v(0), y = v(1), z = v(2);
  const Scalar reference =
      std::max({1.0, std::abs(x), std::abs(y), std::abs(z)});
  if (y > tolerance * reference) {
    if (z <= 0.0) return false;
    return std::log(y) + x / y <= std::log(z) + tolerance;
  }
  return std::abs(y) <= tolerance * reference && x <= tolerance * reference &&
         z >= -tolerance * reference;
}

bool inside_dual_closed(const Vector3& v,
                        Scalar tolerance = kMembershipTolerance) {
  const Scalar u = v(0), w = v(1), t = v(2);
  const Scalar reference =
      std::max({1.0, std::abs(u), std::abs(w), std::abs(t)});
  if (u < -tolerance * reference) {
    if (t <= 0.0) return false;
    return std::log(-u) + w / u <= std::log(t) + 1.0 + tolerance;
  }
  return std::abs(u) <= tolerance * reference &&
         w >= -tolerance * reference && t >= -tolerance * reference;
}

Scalar boundary_equation(Scalar ratio, Scalar exp_minus_ratio, Scalar point_x,
                         Scalar point_y, Scalar point_z) {
  const Scalar coefficient =
      (point_x - ratio * point_y) / (ratio * ratio - ratio + 1.0);
  return coefficient * (exp_minus_ratio * exp_minus_ratio + 1.0 - ratio) -
         point_y + point_z * exp_minus_ratio;
}

Vector3 boundary_point(Scalar ratio, Scalar point_x, Scalar point_y) {
  const Scalar coefficient =
      (point_x - ratio * point_y) / (ratio * ratio - ratio + 1.0);
  const Scalar y = point_y - coefficient * (1.0 - ratio);
  return Vector3(ratio * y, y, y * std::exp(ratio));
}

Vector3 project_onto_cone(const Vector3& point) {
  const Scalar point_x = point(0), point_y = point(1), point_z = point(2);
  if (inside_closed(point)) return point;
  if (inside_dual_closed(-point)) return Vector3::Zero();

  Vector3 best = Vector3::Zero();
  Scalar best_distance = std::numeric_limits<Scalar>::infinity();
  const auto consider = [&](const Vector3& candidate) {
    if (!candidate.allFinite() || !inside_closed(candidate)) return;
    const Scalar distance = (candidate - point).squaredNorm();
    if (distance < best_distance) {
      best_distance = distance;
      best = candidate;
    }
  };
  consider(Vector3::Zero());
  consider(Vector3(std::min(point_x, 0.0), 0.0, std::max(point_z, 0.0)));
  if (point_y > 0.0) {
    const Scalar ratio = point_x / point_y;
    if (ratio < 100.0) {
      const Scalar smallest_z = point_y * std::exp(ratio);
      if (std::isfinite(smallest_z) && std::abs(smallest_z) < 1e100)
        consider(Vector3(point_x, point_y, std::max(point_z, smallest_z)));
    }
  }

  const Scalar step_ratio =
      std::exp(-(kGridUpper - kGridLower) / (kGridPoints - 1));
  Scalar exp_minus_ratio = std::exp(-kGridLower);
  Scalar previous_ratio = kGridLower;
  Scalar previous_value =
      boundary_equation(kGridLower, exp_minus_ratio, point_x, point_y, point_z);
  for (int index = 1; index < kGridPoints; ++index) {
    const Scalar ratio =
        kGridLower + (kGridUpper - kGridLower) * index / (kGridPoints - 1);
    exp_minus_ratio = (index % kAnchorEvery == 0)
                          ? std::exp(-ratio)
                          : exp_minus_ratio * step_ratio;
    const Scalar equation_value =
        boundary_equation(ratio, exp_minus_ratio, point_x, point_y, point_z);
    if (std::isfinite(previous_value) && std::isfinite(equation_value) &&
        previous_value != 0.0 && previous_value * equation_value < 0.0) {
      Scalar lower = previous_ratio;
      Scalar upper = ratio;
      Scalar lower_value = previous_value;
      for (int refinement = 0; refinement < 200; ++refinement) {
        const Scalar middle = 0.5 * (lower + upper);
        const Scalar middle_value = boundary_equation(
            middle, std::exp(-middle), point_x, point_y, point_z);
        if (!std::isfinite(middle_value)) break;
        if (lower_value * middle_value <= 0.0) {
          upper = middle;
        } else {
          lower = middle;
          lower_value = middle_value;
        }
        if (upper - lower < kRootTolerance * std::max(1.0, std::abs(lower)))
          break;
      }
      const Vector3 candidate =
          boundary_point(0.5 * (lower + upper), point_x, point_y);
      if (candidate.allFinite() && candidate(1) >= -kMembershipTolerance)
        consider(candidate);
    }
    previous_ratio = ratio;
    previous_value = equation_value;
  }
  return best;
}

Matrix3 projection_jacobian(const Vector3& point) {
  if (inside_closed(point)) return Matrix3::Identity();
  if (inside_dual_closed(-point)) return Matrix3::Zero();
  const Vector3 projected = project_onto_cone(point);
  const Scalar x = projected(0), y = projected(1);
  if (y <= kDegenerate) {
    Matrix3 jacobian = Matrix3::Zero();
    if (point(0) <= 0.0) jacobian(0, 0) = 1.0;
    if (point(2) >= 0.0) jacobian(2, 2) = 1.0;
    return jacobian;
  }
  const Scalar ratio = x / y;
  const Scalar exponential_value = std::exp(ratio);
  const Vector3 gradient(exponential_value,
                         exponential_value * (1.0 - ratio), -1.0);
  const Scalar multiplier =
      (point - projected).dot(gradient) / gradient.dot(gradient);
  const Vector3 rank_one_direction(1.0, -ratio, 0.0);
  const Matrix3 hessian =
      (exponential_value / y) *
      (rank_one_direction * rank_one_direction.transpose());
  Eigen::Matrix<Scalar, 4, 4> system = Eigen::Matrix<Scalar, 4, 4>::Zero();
  system.topLeftCorner<3, 3>() = Matrix3::Identity() + multiplier * hessian;
  system.topRightCorner<3, 1>() = gradient;
  system.bottomLeftCorner<1, 3>() = gradient.transpose();
  const Eigen::Matrix<Scalar, 4, 4> inverse_system =
      system.fullPivLu().solve(Eigen::Matrix<Scalar, 4, 4>::Identity());
  return inverse_system.topLeftCorner<3, 3>();
}

void barrier_derivatives(const Vector3& s, Vector3& gradient,
                         Matrix3& hessian) {
  const Scalar x = s(0), y = s(1), z = s(2);
  const Scalar slack = y * std::log(z / y) - x;
  const Vector3 slack_gradient(-1.0, std::log(z / y) - 1.0, y / z);
  Matrix3 slack_hessian = Matrix3::Zero();
  slack_hessian(1, 1) = -1.0 / y;
  slack_hessian(1, 2) = slack_hessian(2, 1) = 1.0 / z;
  slack_hessian(2, 2) = -y / (z * z);

  gradient = -slack_gradient / slack;
  gradient(1) -= 1.0 / y;
  gradient(2) -= 1.0 / z;

  hessian = (slack_gradient * slack_gradient.transpose()) / (slack * slack) -
            slack_hessian / slack;
  hessian(1, 1) += 1.0 / (y * y);
  hessian(2, 2) += 1.0 / (z * z);
}

Vector3 primal_direction() { return Vector3(-1.0, 1.0, 2.0); }
Vector3 dual_direction() { return Vector3(-1.0, 0.0, 1.0); }

}

Index rows_impl(const Exponential&) { return 3; }

std::size_t degree_impl(const Exponential&) { return 3; }

bool is_curved_impl(const Exponential&) { return true; }

bool has_interior_impl(const Exponential&) { return true; }

bool contains_impl(const Exponential&, ConeSide side,
                   const ConstVectorRef& v) {
  const Vector3 point(v(0), v(1), v(2));
  return side == ConeSide::Dual ? inside_dual_closed(point)
                                : inside_closed(point);
}

void evaluate_projection_impl(const Exponential&, const ConstVectorRef& v,
                              VectorRef out) {
  const Vector3 point(v(0), v(1), v(2));
  out = point + project_onto_cone(-point);
}

void build_projection_scaling_impl(const Exponential&,
                                   const ConstVectorRef& sigma,
                                   const ConstVectorRef& mu,
                                   BlockScaling& scaling) {
  const Vector3 point(sigma(0), sigma(1), sigma(2));
  scaling.source = ScalingSource::Projection;
  scaling.projection_jacobian =
      Matrix3::Identity() - projection_jacobian(-point);
  scaling.penalty = mu;
}

void build_barrier_scaling_impl(const Exponential&, const ConstVectorRef& s,
                                const ConstVectorRef& z,
                                BlockScaling& scaling) {
  const Vector3 point(s(0), s(1), s(2));
  Vector3 gradient;
  Matrix3 hessian;
  barrier_derivatives(point, gradient, hessian);
  scaling.source = ScalingSource::Barrier;
  scaling.barrier_gradient = gradient;
  scaling.barrier_hessian = hessian;
  scaling.dual_point = z;
  scaling.complementarity = s.dot(z) / 3.0;
}

void materialise_operator_impl(const Exponential&, const BlockScaling& scaling,
                               Scalar rho_p, GBlock& block) {
  block.kind = GBlock::Kind::Dense;
  block.dense = scaling.source == ScalingSource::Barrier
                    ? (scaling.complementarity * scaling.barrier_hessian).eval()
                    : (scaling.projection_jacobian / scaling.penalty(0)).eval();
  block.dense.diagonal().array() += rho_p;
}

GBlock::Kind widest_operator_kind_impl(const Exponential&) {
  return GBlock::Kind::Dense;
}

SchurAssembly schur_assembly_impl(const Exponential&) {
  return SchurAssembly::Materialise;
}

void apply_inverse_operator_impl(const Exponential&, const BlockScaling&,
                                 Scalar, const ConstVectorRef&, VectorRef) {
  schur_weight_is_materialised();
}

void apply_schur_weight_impl(const Exponential&, const BlockScaling&, Scalar,
                             const ConstVectorRef&, const ConstVectorRef&,
                             VectorRef) {
  schur_weight_is_materialised();
}

void apply_residual_scaling_impl(const Exponential&,
                                 const BlockScaling& scaling,
                                 const ConstVectorRef& v, VectorRef out) {
  if (scaling.source == ScalingSource::Projection) {
    out = scaling.projection_jacobian * v;
    return;
  }
  out = v;
}

void complementarity_residual_impl(const Exponential&,
                                   const BlockScaling& scaling, Scalar target,
                                   VectorRef out) {
  out = scaling.dual_point + target * scaling.barrier_gradient;
}

void second_order_correction_impl(const Exponential&, const BlockScaling&,
                                  const ConstVectorRef&, const ConstVectorRef&,
                                  VectorRef out) {
  out.setZero();
}

Scalar largest_feasible_step_impl(const Exponential& cone, ConeSide side,
                                  const ConstVectorRef& v,
                                  const ConstVectorRef& dv) {
  return side == ConeSide::Dual
             ? bisect_step(cone, v, dv, inside_dual_interior)
             : bisect_step(cone, v, dv, inside_interior);
}

void interior_direction_impl(const Exponential&, ConeSide side,
                             VectorRef out) {
  out = side == ConeSide::Dual ? dual_direction() : primal_direction();
}

Scalar margin_impl(const Exponential& cone, ConeSide side,
                   const ConstVectorRef& v) {
  return side == ConeSide::Dual
             ? bisect_margin(cone, v, dual_direction(),
                             [](const Exponential&, const Vector3& point) {
                               return inside_dual_closed(point);
                             })
             : bisect_margin(cone, v, primal_direction(),
                             [](const Exponential&, const Vector3& point) {
                               return inside_closed(point);
                             });
}

}
