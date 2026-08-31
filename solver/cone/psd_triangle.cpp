#include <algorithm>
#include <cmath>
#include <limits>

#include <Eigen/Eigenvalues>

#include "cone/kernel.hpp"
#include "cone/svec.hpp"

namespace proxgqp::detail {
namespace {

constexpr Scalar kMembershipTolerance = 1e-9;
constexpr Scalar kEigenvalueFloor = 1e-14;

template <typename Function>
DenseMatrix apply_to_eigenvalues(const DenseMatrix& matrix, Function function) {
  Eigen::SelfAdjointEigenSolver<DenseMatrix> eigen(matrix);
  Vector mapped = eigen.eigenvalues();
  const Scalar floor =
      kEigenvalueFloor * std::max(Scalar(1), mapped.cwiseAbs().maxCoeff());
  for (Index index = 0; index < mapped.size(); ++index)
    mapped(index) = function(std::max(mapped(index), floor));
  return eigen.eigenvectors() * mapped.asDiagonal() *
         eigen.eigenvectors().transpose();
}

void matrix_square_roots(const DenseMatrix& matrix, DenseMatrix& root,
                         DenseMatrix& inverse_root) {
  Eigen::SelfAdjointEigenSolver<DenseMatrix> eigen(matrix);
  const Vector eigenvalues = eigen.eigenvalues();
  const Scalar floor =
      kEigenvalueFloor * std::max(Scalar(1), eigenvalues.cwiseAbs().maxCoeff());
  Vector roots(eigenvalues.size());
  Vector inverse_roots(eigenvalues.size());
  for (Index index = 0; index < eigenvalues.size(); ++index) {
    roots(index) = std::sqrt(std::max(eigenvalues(index), floor));
    inverse_roots(index) = 1.0 / roots(index);
  }
  const DenseMatrix& basis = eigen.eigenvectors();
  root.noalias() = basis * roots.asDiagonal() * basis.transpose();
  inverse_root.noalias() =
      basis * inverse_roots.asDiagonal() * basis.transpose();
}

DenseMatrix divided_differences(const Vector& eigenvalues) {
  const Index side = eigenvalues.size();
  DenseMatrix weights(side, side);
  for (Index row = 0; row < side; ++row)
    for (Index column = 0; column < side; ++column) {
      const Scalar gap = eigenvalues(row) - eigenvalues(column);
      const Scalar reference = std::max({std::abs(eigenvalues(row)),
                                         std::abs(eigenvalues(column)), 1.0});
      weights(row, column) =
          (std::abs(gap) > std::numeric_limits<Scalar>::epsilon() * reference)
              ? (std::max(eigenvalues(row), 0.0) -
                 std::max(eigenvalues(column), 0.0)) /
                    gap
              : (eigenvalues(row) > 0.0 ? 1.0 : 0.0);
    }
  return weights;
}

void apply_cached_projection_jacobian(
    const PSDTriangle& cone, const DenseMatrix& eigenvectors,
    const DenseMatrix& divided_differences, const ConstVectorRef& vector,
    VectorRef out, DenseMatrix& work_matrix, DenseMatrix& eigenbasis_matrix) {
  matrix_from_svec(cone, vector, work_matrix);
  eigenbasis_matrix.noalias() =
      eigenvectors.transpose() * work_matrix * eigenvectors;
  eigenbasis_matrix.array() *= divided_differences.array();
  work_matrix.noalias() =
      eigenvectors * eigenbasis_matrix * eigenvectors.transpose();
  svec_from_matrix(cone, work_matrix, out);
}

template <typename Weight>
void apply_in_operator_basis(const PSDTriangle& cone,
                             const BlockScaling& scaling, Scalar rho_p,
                             const ConstVectorRef& v, VectorRef out,
                             Weight weight) {
  const Index side = static_cast<Index>(cone.side);
  const bool barrier = scaling.source == ScalingSource::Barrier;
  const DenseMatrix& basis =
      barrier ? scaling.congruence_eigenvectors : scaling.eigenvectors;
  DenseMatrix work_matrix;
  matrix_from_svec(cone, v, work_matrix);
  DenseMatrix in_basis = basis.transpose() * work_matrix * basis;
  for (Index column = 0; column < side; ++column)
    for (Index row = 0; row < side; ++row) {
      const Scalar operator_eigenvalue =
          barrier ? scaling.congruence_eigenvalues(row) *
                        scaling.congruence_eigenvalues(column)
                  : scaling.divided_differences(row, column) /
                        scaling.penalty(0);
      in_basis(row, column) *= weight(rho_p + operator_eigenvalue);
    }
  work_matrix.noalias() = basis * in_basis * basis.transpose();
  svec_from_matrix(cone, work_matrix, out);
}

}

Index rows_impl(const PSDTriangle& cone) {
  return static_cast<Index>(cone.dim);
}

std::size_t degree_impl(const PSDTriangle& cone) { return cone.side; }

bool is_curved_impl(const PSDTriangle&) { return true; }

bool has_interior_impl(const PSDTriangle&) { return true; }

bool contains_impl(const PSDTriangle& cone, ConeSide,
                   const ConstVectorRef& v) {
  DenseMatrix matrix;
  matrix_from_svec(cone, v, matrix);
  const Scalar reference = std::max(1.0, matrix.cwiseAbs().maxCoeff());
  matrix.diagonal().array() += kMembershipTolerance * reference;
  Eigen::LLT<DenseMatrix> cholesky(matrix);
  return cholesky.info() == Eigen::Success;
}

void evaluate_projection_impl(const PSDTriangle& cone, const ConstVectorRef& v,
                              VectorRef out) {
  DenseMatrix matrix;
  matrix_from_svec(cone, v, matrix);
  Eigen::SelfAdjointEigenSolver<DenseMatrix> eigen(matrix);
  const Vector clipped = eigen.eigenvalues().cwiseMax(0.0);
  const DenseMatrix projected =
      eigen.eigenvectors() * clipped.asDiagonal() *
      eigen.eigenvectors().transpose();
  svec_from_matrix(cone, projected, out);
}

void build_projection_scaling_impl(const PSDTriangle& cone,
                                   const ConstVectorRef& sigma,
                                   const ConstVectorRef& mu,
                                   BlockScaling& scaling) {
  DenseMatrix matrix;
  matrix_from_svec(cone, sigma, matrix);
  Eigen::SelfAdjointEigenSolver<DenseMatrix> eigen(matrix);
  scaling.source = ScalingSource::Projection;
  scaling.eigenvectors = eigen.eigenvectors();
  scaling.divided_differences = divided_differences(eigen.eigenvalues());
  scaling.penalty = mu;
}

void build_barrier_scaling_impl(const PSDTriangle& cone,
                                const ConstVectorRef& s,
                                const ConstVectorRef& z,
                                BlockScaling& scaling) {
  DenseMatrix primal_matrix;
  DenseMatrix dual_matrix;
  matrix_from_svec(cone, s, primal_matrix);
  matrix_from_svec(cone, z, dual_matrix);

  DenseMatrix root_primal;
  DenseMatrix inverse_root_primal;
  matrix_square_roots(primal_matrix, root_primal, inverse_root_primal);
  const DenseMatrix inner_root = apply_to_eigenvalues(
      (root_primal * dual_matrix * root_primal).eval(),
      [](Scalar eigenvalue) { return std::sqrt(eigenvalue); });
  const DenseMatrix congruence =
      inverse_root_primal * inner_root * inverse_root_primal;

  scaling.source = ScalingSource::Barrier;
  Eigen::SelfAdjointEigenSolver<DenseMatrix> congruence_eigen(congruence);
  scaling.congruence_eigenvectors = congruence_eigen.eigenvectors();
  scaling.congruence_eigenvalues = congruence_eigen.eigenvalues();

  const Index side = static_cast<Index>(cone.side);
  Vector roots(side);
  Vector inverse_roots(side);
  for (Index index = 0; index < side; ++index) {
    roots(index) = std::sqrt(scaling.congruence_eigenvalues(index));
    inverse_roots(index) = 1.0 / roots(index);
  }
  const DenseMatrix& congruence_basis = scaling.congruence_eigenvectors;
  scaling.congruence_root.noalias() =
      congruence_basis * roots.asDiagonal() * congruence_basis.transpose();
  scaling.inverse_congruence_root.noalias() =
      congruence_basis * inverse_roots.asDiagonal() *
      congruence_basis.transpose();

  DenseMatrix lambda_matrix =
      scaling.congruence_root * primal_matrix * scaling.congruence_root;
  lambda_matrix = 0.5 * (lambda_matrix + lambda_matrix.transpose()).eval();
  Eigen::SelfAdjointEigenSolver<DenseMatrix> eigen(lambda_matrix);
  scaling.lambda = eigen.eigenvalues();
  scaling.eigenvectors = eigen.eigenvectors();
}

void materialise_operator_impl(const PSDTriangle& cone,
                               const BlockScaling& scaling, Scalar rho_p,
                               GBlock& block) {
  const Index side = static_cast<Index>(cone.side);
  const Index dim = static_cast<Index>(cone.dim);
  block.kind = GBlock::Kind::Dense;
  block.dense.resize(dim, dim);

  if (scaling.source == ScalingSource::Projection) {
    DenseMatrix work_matrix;
    DenseMatrix eigenbasis_matrix;
    Vector unit_vector = Vector::Zero(dim);
    Vector column(dim);
    for (Index index = 0; index < dim; ++index) {
      unit_vector(index) = 1.0;
      apply_cached_projection_jacobian(cone, scaling.eigenvectors,
                                       scaling.divided_differences,
                                       unit_vector, column, work_matrix,
                                       eigenbasis_matrix);
      block.dense.col(index) = column / scaling.penalty(0);
      unit_vector(index) = 0.0;
    }
    block.dense.diagonal().array() += rho_p;
    return;
  }

  const DenseMatrix& congruence_basis = scaling.congruence_eigenvectors;
  const DenseMatrix congruence = congruence_basis *
                                 scaling.congruence_eigenvalues.asDiagonal() *
                                 congruence_basis.transpose();
  DenseMatrix outer_product(side, side);
  Index position = 0;
  for (Index column = 0; column < side; ++column)
    for (Index row = column; row < side; ++row, ++position) {
      if (row == column) {
        outer_product.noalias() =
            congruence.col(row) * congruence.col(row).transpose();
      } else {
        outer_product.noalias() =
            congruence.col(row) * congruence.col(column).transpose();
        outer_product += outer_product.transpose().eval();
        outer_product *= 1.0 / std::sqrt(2.0);
      }
      svec_from_matrix(cone, outer_product, block.dense.col(position));
    }
  block.dense.diagonal().array() += rho_p;
}

GBlock::Kind widest_operator_kind_impl(const PSDTriangle&) {
  return GBlock::Kind::Dense;
}

SchurAssembly schur_assembly_impl(const PSDTriangle&) {
  return SchurAssembly::ByApplication;
}

void apply_inverse_operator_impl(const PSDTriangle& cone,
                                 const BlockScaling& scaling, Scalar rho_p,
                                 const ConstVectorRef& v, VectorRef out) {
  apply_in_operator_basis(cone, scaling, rho_p, v, out,
                          [](Scalar entry) { return 1.0 / entry; });
}

void apply_schur_weight_impl(const PSDTriangle& cone,
                             const BlockScaling& scaling, Scalar rho_p,
                             const ConstVectorRef& mu_s,
                             const ConstVectorRef& v, VectorRef out) {
  const Scalar equality_penalty = mu_s(0);
  apply_in_operator_basis(
      cone, scaling, rho_p, v, out, [equality_penalty](Scalar entry) {
        return 1.0 / (1.0 / entry + equality_penalty);
      });
}

void apply_residual_scaling_impl(const PSDTriangle& cone,
                                 const BlockScaling& scaling,
                                 const ConstVectorRef& v, VectorRef out) {
  const Index side = static_cast<Index>(cone.side);
  if (scaling.source == ScalingSource::Projection) {
    DenseMatrix work_matrix;
    DenseMatrix eigenbasis_matrix;
    apply_cached_projection_jacobian(cone, scaling.eigenvectors,
                                     scaling.divided_differences, v, out,
                                     work_matrix, eigenbasis_matrix);
    return;
  }
  DenseMatrix residual_matrix;
  matrix_from_svec(cone, v, residual_matrix);
  DenseMatrix in_eigenbasis = scaling.eigenvectors.transpose() *
                              residual_matrix * scaling.eigenvectors;
  for (Index column = 0; column < side; ++column)
    for (Index row = 0; row < side; ++row)
      in_eigenbasis(row, column) *=
          2.0 / (scaling.lambda(row) + scaling.lambda(column));
  DenseMatrix lyapunov_solution =
      scaling.eigenvectors * in_eigenbasis * scaling.eigenvectors.transpose();
  const DenseMatrix congruence_image =
      scaling.congruence_root * lyapunov_solution * scaling.congruence_root;
  svec_from_matrix(cone, congruence_image, out);
}

void complementarity_residual_impl(const PSDTriangle& cone,
                                   const BlockScaling& scaling, Scalar target,
                                   VectorRef out) {
  const Vector diagonal = scaling.lambda.array().square() - target;
  const DenseMatrix residual =
      scaling.eigenvectors * diagonal.asDiagonal() *
      scaling.eigenvectors.transpose();
  svec_from_matrix(cone, residual, out);
}

void second_order_correction_impl(const PSDTriangle& cone,
                                  const BlockScaling& scaling,
                                  const ConstVectorRef& ds,
                                  const ConstVectorRef& dz, VectorRef out) {
  DenseMatrix primal_step;
  DenseMatrix dual_step;
  matrix_from_svec(cone, ds, primal_step);
  matrix_from_svec(cone, dz, dual_step);
  const DenseMatrix scaled_primal =
      scaling.congruence_root * primal_step * scaling.congruence_root;
  const DenseMatrix scaled_dual =
      scaling.inverse_congruence_root * dual_step *
      scaling.inverse_congruence_root;
  const DenseMatrix product =
      0.5 * (scaled_primal * scaled_dual + scaled_dual * scaled_primal);
  svec_from_matrix(cone, product, out);
}

Scalar largest_feasible_step_impl(const PSDTriangle& cone, ConeSide,
                                  const ConstVectorRef& v,
                                  const ConstVectorRef& dv) {
  DenseMatrix current;
  DenseMatrix step;
  matrix_from_svec(cone, v, current);
  matrix_from_svec(cone, dv, step);
  const DenseMatrix inverse_root =
      apply_to_eigenvalues(current, [](Scalar eigenvalue) {
        return 1.0 / std::sqrt(eigenvalue);
      });
  DenseMatrix transformed = inverse_root * step * inverse_root;
  transformed = 0.5 * (transformed + transformed.transpose()).eval();
  Eigen::SelfAdjointEigenSolver<DenseMatrix> eigen(transformed,
                                                   Eigen::EigenvaluesOnly);
  const Scalar smallest_eigenvalue = eigen.eigenvalues().minCoeff();
  return smallest_eigenvalue < 0.0 ? -1.0 / smallest_eigenvalue
                        : std::numeric_limits<Scalar>::infinity();
}

void interior_direction_impl(const PSDTriangle& cone, ConeSide,
                             VectorRef out) {
  const Index side = static_cast<Index>(cone.side);
  out.setZero();
  Index position = 0;
  for (Index column = 0; column < side; ++column)
    for (Index row = column; row < side; ++row, ++position)
      if (row == column) out(position) = 1.0;
}

Scalar margin_impl(const PSDTriangle& cone, ConeSide, const ConstVectorRef& v) {
  DenseMatrix matrix;
  matrix_from_svec(cone, v, matrix);
  Eigen::SelfAdjointEigenSolver<DenseMatrix> eigen(matrix,
                                                   Eigen::EigenvaluesOnly);
  return eigen.eigenvalues().minCoeff();
}

}
