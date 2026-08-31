#include "ruiz/ruiz.hpp"

#include <algorithm>
#include <cmath>

#include "cone/kernel.hpp"

namespace proxgqp {
namespace {

constexpr Scalar kEmpty = 1e-12;
constexpr Scalar kSmallest = 1e-8;
constexpr Scalar kLargest = 1e8;

Scalar guarded_root(Scalar value) {
  if (!(value > kEmpty)) return 1.0;
  const Scalar clamped = std::min(kLargest, std::max(kSmallest, value));
  return 1.0 / std::sqrt(clamped);
}

void flatten_curved(const Cones& cones, const std::vector<Index>& offsets,
                    Vector& row_pass) {
  for (std::size_t j = 0; j < cones.size(); ++j) {
    if (!is_curved(cones[j])) continue;
    const Index start = offsets[j];
    const Index length = offsets[j + 1] - offsets[j];
    const Scalar shared =
        std::exp(row_pass.segment(start, length).array().log().mean());
    row_pass.segment(start, length).setConstant(shared);
  }
}

}

void equilibrate(const SparseMatrix& P, const Vector& q, const SparseMatrix& E,
                 const Vector& b, const Cones& cones,
                 const RuizSettings& settings, ScaledProblem& scaled) {
  const Index columns = P.rows();
  const Index rows = E.rows();
  const std::vector<Index> offsets = block_offsets(cones);

  scaled.P = P;
  scaled.E = E;
  scaled.q = q;
  scaled.b = b;
  scaled.column_scale.setOnes(columns);
  scaled.row_scale.setOnes(rows);
  scaled.cost_scale = 1.0;

  Vector column_pass(columns);
  Vector row_pass(rows);
  for (std::size_t pass = 0; pass < settings.passes; ++pass) {
    column_pass.setZero();
    row_pass.setZero();
    for (Index column = 0; column < columns; ++column)
      for (SparseMatrix::InnerIterator it(scaled.P, column); it; ++it)
        column_pass(column) =
            std::max(column_pass(column), std::abs(it.value()));
    for (Index column = 0; column < columns; ++column)
      for (SparseMatrix::InnerIterator it(scaled.E, column); it; ++it) {
        column_pass(column) =
            std::max(column_pass(column), std::abs(it.value()));
        row_pass(it.row()) = std::max(row_pass(it.row()), std::abs(it.value()));
      }
    Scalar worst = 0.0;
    for (Index column = 0; column < columns; ++column)
      if (column_pass(column) > kEmpty)
        worst = std::max(worst, std::abs(column_pass(column) - 1.0));
    for (Index row = 0; row < rows; ++row)
      if (row_pass(row) > kEmpty)
        worst = std::max(worst, std::abs(row_pass(row) - 1.0));

    for (Index column = 0; column < columns; ++column)
      column_pass(column) = guarded_root(column_pass(column));
    for (Index row = 0; row < rows; ++row)
      row_pass(row) = guarded_root(row_pass(row));
    flatten_curved(cones, offsets, row_pass);

    for (Index column = 0; column < columns; ++column)
      for (SparseMatrix::InnerIterator it(scaled.P, column); it; ++it)
        it.valueRef() *= column_pass(column) * column_pass(it.row());
    for (Index column = 0; column < columns; ++column)
      for (SparseMatrix::InnerIterator it(scaled.E, column); it; ++it)
        it.valueRef() *= row_pass(it.row()) * column_pass(column);
    scaled.q.array() *= column_pass.array();
    scaled.b.array() *= row_pass.array();
    scaled.column_scale.array() *= column_pass.array();
    scaled.row_scale.array() *= row_pass.array();
    if (worst <= settings.tolerance) break;
  }

  if (!settings.scale_cost) return;
  Scalar objective_norm = 0.0;
  for (Index column = 0; column < columns; ++column)
    for (SparseMatrix::InnerIterator it(scaled.P, column); it; ++it)
      objective_norm = std::max(objective_norm, std::abs(it.value()));
  const Scalar linear_norm =
      scaled.q.size() ? scaled.q.lpNorm<Eigen::Infinity>() : 0.0;
  const Scalar reference = std::max(objective_norm, linear_norm);
  if (reference <= kSmallest || reference >= kLargest) return;
  scaled.cost_scale = 1.0 / reference;
  scaled.P *= scaled.cost_scale;
  scaled.q *= scaled.cost_scale;
}

void unscale_primal(const ScaledProblem& scaled, const Vector& x_scaled,
                    Vector& x) {
  x = scaled.column_scale.cwiseProduct(x_scaled);
}

void unscale_slack(const ScaledProblem& scaled, const Vector& s_scaled,
                   Vector& s) {
  s = s_scaled.cwiseQuotient(scaled.row_scale);
}

void unscale_dual(const ScaledProblem& scaled, const Vector& z_scaled,
                  Vector& z) {
  z = scaled.row_scale.cwiseProduct(z_scaled) / scaled.cost_scale;
}

void scale_primal(const ScaledProblem& scaled, const Vector& x, Vector& out) {
  out = x.cwiseQuotient(scaled.column_scale);
}

void scale_slack(const ScaledProblem& scaled, const Vector& s, Vector& out) {
  out = scaled.row_scale.cwiseProduct(s);
}

void scale_dual(const ScaledProblem& scaled, const Vector& z, Vector& out) {
  out = scaled.cost_scale * z.cwiseQuotient(scaled.row_scale);
}

}
