#include <algorithm>
#include <cmath>
#include <limits>

#include <Eigen/LU>

#include "cone/curved.hpp"
#include "cone/kernel.hpp"

namespace proxgqp::detail {
namespace {

constexpr Scalar kMembershipTolerance = 1e-9;
constexpr Scalar kCandidateTolerance = 1e-8;
constexpr Scalar kDegenerate = 1e-13;
constexpr int kBisectionRefinements = 400;

bool inside_closed(const Vector3& v, Scalar alpha,
                   Scalar tolerance = kMembershipTolerance) {
  Scalar x = v(0), y = v(1);
  const Scalar z = v(2);
  const Scalar reference =
      std::max({1.0, std::abs(x), std::abs(y), std::abs(z)});
  if (x < -tolerance * reference || y < -tolerance * reference) return false;
  x = std::max(x, 0.0);
  y = std::max(y, 0.0);
  if (x == 0.0 || y == 0.0) return std::abs(z) <= tolerance * reference;
  if (std::abs(z) == 0.0) return true;
  return alpha * std::log(x) + (1.0 - alpha) * std::log(y) >=
         std::log(std::abs(z)) - tolerance;
}

bool inside_dual_closed(const Vector3& v, Scalar alpha,
                        Scalar tolerance = kMembershipTolerance) {
  return inside_closed(Vector3(v(0) / alpha, v(1) / (1.0 - alpha), v(2)), alpha,
                       tolerance);
}

bool inside_interior(const Power& cone, const Vector3& v) {
  const Scalar x = v(0), y = v(1), z = v(2);
  if (!(x > 0.0 && y > 0.0)) return false;
  const Scalar alpha = cone.alpha;
  return std::pow(x, 2.0 * alpha) * std::pow(y, 2.0 * (1.0 - alpha)) - z * z >
         0.0;
}

bool inside_dual_interior(const Power& cone, const Vector3& v) {
  const Scalar alpha = cone.alpha;
  return inside_interior(
      cone, Vector3(v(0) / alpha, v(1) / (1.0 - alpha), v(2)));
}

Scalar positive_root(Scalar linear, Scalar product) {
  if (product <= 0.0) return std::max(linear, 0.0);
  const Scalar discriminant_root =
      std::sqrt(linear * linear + 4.0 * product);
  return (linear >= 0.0) ? 0.5 * (linear + discriminant_root)
                         : 2.0 * product / (discriminant_root - linear);
}

struct BoundaryValue {
  Scalar value;
  Scalar x;
  Scalar y;
  Scalar z;
};

BoundaryValue boundary_value(Scalar multiplier, Scalar point_x, Scalar point_y,
                             Scalar absolute_z, Scalar alpha) {
  const Scalar z = absolute_z - multiplier;
  const Scalar x = positive_root(point_x, multiplier * alpha * z);
  const Scalar y = positive_root(point_y, multiplier * (1.0 - alpha) * z);
  if (x <= 0.0 || y <= 0.0) return {-z, x, y, z};
  return {std::exp(alpha * std::log(x) + (1.0 - alpha) * std::log(y)) - z, x, y,
          z};
}

Vector3 project_onto_cone(const Vector3& point, Scalar alpha) {
  if (inside_closed(point, alpha)) return point;
  if (inside_dual_closed(-point, alpha)) return Vector3::Zero();
  const Scalar point_x = point(0), point_y = point(1), point_z = point(2);

  Vector3 best = Vector3::Zero();
  Scalar best_distance = point.squaredNorm();
  const auto consider = [&](const Vector3& candidate) {
    if (!candidate.allFinite() ||
        !inside_closed(candidate, alpha, kCandidateTolerance))
      return;
    const Scalar distance = (candidate - point).squaredNorm();
    if (distance < best_distance) {
      best_distance = distance;
      best = candidate;
    }
  };
  consider(Vector3(0.0, std::max(point_y, 0.0), 0.0));
  consider(Vector3(std::max(point_x, 0.0), 0.0, 0.0));
  consider(Vector3(std::max(point_x, 0.0), std::max(point_y, 0.0), 0.0));

  const Scalar sign = (point_z >= 0.0) ? 1.0 : -1.0;
  const Scalar absolute_z = std::abs(point_z);
  if (absolute_z > 0.0) {
    Scalar lower = 0.0;
    Scalar upper = absolute_z;
    Scalar lower_value =
        boundary_value(lower, point_x, point_y, absolute_z, alpha).value;
    const Scalar upper_value =
        boundary_value(upper, point_x, point_y, absolute_z, alpha).value;
    if (std::isfinite(lower_value) && std::isfinite(upper_value) &&
        (lower_value < 0.0) != (upper_value < 0.0)) {
      for (int refinement = 0; refinement < kBisectionRefinements;
           ++refinement) {
        const Scalar middle = 0.5 * (lower + upper);
        const Scalar middle_value =
            boundary_value(middle, point_x, point_y, absolute_z, alpha).value;
        if ((lower_value < 0.0) != (middle_value < 0.0)) {
          upper = middle;
        } else {
          lower = middle;
          lower_value = middle_value;
        }
        if (upper - lower <= 1e-15 * std::max(upper, 1e-300)) break;
      }
      const BoundaryValue boundary_solution = boundary_value(
          0.5 * (lower + upper), point_x, point_y, absolute_z, alpha);
      consider(Vector3(std::max(boundary_solution.x, 0.0),
                       std::max(boundary_solution.y, 0.0),
                       sign * std::max(boundary_solution.z, 0.0)));
    }
  }
  return best;
}

Matrix3 projection_jacobian(const Vector3& point, Scalar alpha) {
  if (inside_closed(point, alpha)) return Matrix3::Identity();
  if (inside_dual_closed(-point, alpha)) return Matrix3::Zero();
  const Vector3 projected = project_onto_cone(point, alpha);
  const Scalar x = projected(0), y = projected(1), z = projected(2);

  if (x <= kDegenerate || y <= kDegenerate || std::abs(z) <= kDegenerate) {
    Matrix3 jacobian = Matrix3::Zero();
    if (x > kDegenerate && point(0) > 0.0) jacobian(0, 0) = 1.0;
    if (y > kDegenerate && point(1) > 0.0) jacobian(1, 1) = 1.0;
    return jacobian;
  }

  const Scalar geometric_mean =
      std::exp(alpha * std::log(x) + (1.0 - alpha) * std::log(y));
  const Scalar sign = (z >= 0.0) ? 1.0 : -1.0;
  const Vector3 gradient(alpha * geometric_mean / x,
                         (1.0 - alpha) * geometric_mean / y, -sign);
  const Scalar multiplier =
      (point - projected).dot(gradient) / gradient.dot(gradient);
  const Vector3 rank_one_direction(1.0 / x, -1.0 / y, 0.0);
  const Matrix3 hessian =
      -alpha * (1.0 - alpha) * geometric_mean *
      (rank_one_direction * rank_one_direction.transpose());

  Eigen::Matrix<Scalar, 4, 4> system = Eigen::Matrix<Scalar, 4, 4>::Zero();
  system.topLeftCorner<3, 3>() = Matrix3::Identity() + multiplier * hessian;
  system.topRightCorner<3, 1>() = gradient;
  system.bottomLeftCorner<1, 3>() = gradient.transpose();
  const Eigen::Matrix<Scalar, 4, 4> inverse_system =
      system.fullPivLu().solve(Eigen::Matrix<Scalar, 4, 4>::Identity());
  const Matrix3 jacobian = inverse_system.topLeftCorner<3, 3>();
  return 0.5 * (jacobian + jacobian.transpose());
}

void barrier_derivatives(const Power& cone, const Vector3& s,
                         Vector3& gradient, Matrix3& hessian) {
  const Scalar x = s(0), y = s(1), z = s(2), alpha = cone.alpha;
  const Scalar product =
      std::pow(x, 2.0 * alpha) * std::pow(y, 2.0 * (1.0 - alpha));
  const Scalar slack = product - z * z;
  const Vector3 slack_gradient(2.0 * alpha * product / x,
                               2.0 * (1.0 - alpha) * product / y, -2.0 * z);
  Matrix3 slack_hessian = Matrix3::Zero();
  slack_hessian(0, 0) = 2.0 * alpha * (2.0 * alpha - 1.0) * product / (x * x);
  slack_hessian(0, 1) = slack_hessian(1, 0) =
      4.0 * alpha * (1.0 - alpha) * product / (x * y);
  slack_hessian(1, 1) =
      2.0 * (1.0 - alpha) * (1.0 - 2.0 * alpha) * product / (y * y);
  slack_hessian(2, 2) = -2.0;

  gradient = -slack_gradient / slack;
  gradient(0) -= (1.0 - alpha) / x;
  gradient(1) -= alpha / y;

  hessian = (slack_gradient * slack_gradient.transpose()) / (slack * slack) -
            slack_hessian / slack;
  hessian(0, 0) += (1.0 - alpha) / (x * x);
  hessian(1, 1) += alpha / (y * y);
}

Vector3 primal_direction() {
  const Scalar norm = std::sqrt(3.0);
  return Vector3(norm, norm, 0.0);
}

Vector3 dual_direction(const Power& cone) {
  const Scalar norm = std::sqrt(3.0);
  return Vector3(norm * cone.alpha, norm * (1.0 - cone.alpha), 0.0);
}

}

Index rows_impl(const Power&) { return 3; }

std::size_t degree_impl(const Power&) { return 3; }

bool is_curved_impl(const Power&) { return true; }

bool has_interior_impl(const Power&) { return true; }

bool contains_impl(const Power& cone, ConeSide side, const ConstVectorRef& v) {
  const Vector3 point(v(0), v(1), v(2));
  return side == ConeSide::Dual ? inside_dual_closed(point, cone.alpha)
                                : inside_closed(point, cone.alpha);
}

void evaluate_projection_impl(const Power& cone, const ConstVectorRef& v,
                              VectorRef out) {
  const Vector3 point(v(0), v(1), v(2));
  out = point + project_onto_cone(-point, cone.alpha);
}

void build_projection_scaling_impl(const Power& cone,
                                   const ConstVectorRef& sigma,
                                   const ConstVectorRef& mu,
                                   BlockScaling& scaling) {
  const Vector3 point(sigma(0), sigma(1), sigma(2));
  scaling.source = ScalingSource::Projection;
  scaling.projection_jacobian =
      Matrix3::Identity() - projection_jacobian(-point, cone.alpha);
  scaling.penalty = mu;
}

void build_barrier_scaling_impl(const Power& cone, const ConstVectorRef& s,
                                const ConstVectorRef& z,
                                BlockScaling& scaling) {
  const Vector3 point(s(0), s(1), s(2));
  Vector3 gradient;
  Matrix3 hessian;
  barrier_derivatives(cone, point, gradient, hessian);
  scaling.source = ScalingSource::Barrier;
  scaling.barrier_gradient = gradient;
  scaling.barrier_hessian = hessian;
  scaling.dual_point = z;
  scaling.complementarity = s.dot(z) / 3.0;
}

void materialise_operator_impl(const Power&, const BlockScaling& scaling,
                               Scalar rho_p, GBlock& block) {
  block.kind = GBlock::Kind::Dense;
  block.dense = scaling.source == ScalingSource::Barrier
                    ? (scaling.complementarity * scaling.barrier_hessian).eval()
                    : (scaling.projection_jacobian / scaling.penalty(0)).eval();
  block.dense.diagonal().array() += rho_p;
}

GBlock::Kind widest_operator_kind_impl(const Power&) {
  return GBlock::Kind::Dense;
}

SchurAssembly schur_assembly_impl(const Power&) {
  return SchurAssembly::Materialise;
}


void apply_schur_weight_impl(const Power&, const BlockScaling&, Scalar,
                             const ConstVectorRef&, const ConstVectorRef&,
                             VectorRef) {
  schur_weight_is_materialised();
}

void apply_residual_scaling_impl(const Power&, const BlockScaling& scaling,
                                 const ConstVectorRef& v, VectorRef out) {
  if (scaling.source == ScalingSource::Projection) {
    out = scaling.projection_jacobian * v;
    return;
  }
  out = v;
}

void complementarity_residual_impl(const Power&, const BlockScaling& scaling,
                                   Scalar target, VectorRef out) {
  out = scaling.dual_point + target * scaling.barrier_gradient;
}

void second_order_correction_impl(const Power&, const BlockScaling&,
                                  const ConstVectorRef&, const ConstVectorRef&,
                                  VectorRef out) {
  out.setZero();
}

Scalar largest_feasible_step_impl(const Power& cone, ConeSide side,
                                  const ConstVectorRef& v,
                                  const ConstVectorRef& dv) {
  return side == ConeSide::Dual
             ? bisect_step(cone, v, dv, inside_dual_interior)
             : bisect_step(cone, v, dv, inside_interior);
}

void interior_direction_impl(const Power& cone, ConeSide side, VectorRef out) {
  out = side == ConeSide::Dual ? dual_direction(cone) : primal_direction();
}

Scalar margin_impl(const Power& cone, ConeSide side, const ConstVectorRef& v) {
  const Scalar alpha = cone.alpha;
  return side == ConeSide::Dual
             ? bisect_margin(cone, v, dual_direction(cone),
                             [alpha](const Power&, const Vector3& point) {
                               return inside_dual_closed(point, alpha);
                             })
             : bisect_margin(cone, v, primal_direction(),
                             [alpha](const Power&, const Vector3& point) {
                               return inside_closed(point, alpha);
                             });
}

}
