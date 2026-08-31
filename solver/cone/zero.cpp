#include <limits>
#include <stdexcept>

#include "cone/kernel.hpp"

namespace proxgqp::detail {
namespace {

[[noreturn]] void no_interior() {
  throw std::domain_error(
      "the zero cone has empty interior, so it carries no interior direction, "
      "no margin and no feasible step");
}

[[noreturn]] void degenerate_barrier() {
  throw std::domain_error(
      "the zero cone's barrier is identically zero on its relative interior, "
      "of degree zero: it smooths nothing, and the row it would generate "
      "drives the multiplier to zero rather than enforcing the equality. A "
      "zero block keeps the projection relation on both roads, with F the "
      "identity and one over mu the penalty");
}

}

Index rows_impl(const Zero& cone) { return static_cast<Index>(cone.dim); }

std::size_t degree_impl(const Zero&) { return 0; }

bool is_curved_impl(const Zero&) { return false; }

bool has_interior_impl(const Zero&) { return false; }

bool contains_impl(const Zero&, ConeSide side, const ConstVectorRef& v) {
  if (side == ConeSide::Dual) return true;
  return v.isZero(0.0);
}

void evaluate_projection_impl(const Zero&, const ConstVectorRef& v,
                              VectorRef out) {
  out = v;
}

void build_projection_scaling_impl(const Zero&, const ConstVectorRef&,
                                   const ConstVectorRef& mu,
                                   BlockScaling& scaling) {
  scaling.source = ScalingSource::Projection;
  scaling.penalty = mu;
}

void build_barrier_scaling_impl(const Zero&, const ConstVectorRef&,
                                const ConstVectorRef&, BlockScaling&) {
  degenerate_barrier();
}

void materialise_operator_impl(const Zero&, const BlockScaling& scaling,
                               Scalar rho_p, GBlock& block) {
  block.kind = GBlock::Kind::Diagonal;
  block.diagonal = rho_p + scaling.penalty.array().inverse();
}

GBlock::Kind widest_operator_kind_impl(const Zero&) {
  return GBlock::Kind::Diagonal;
}

SchurAssembly schur_assembly_impl(const Zero&) {
  return SchurAssembly::Materialise;
}


void apply_schur_weight_impl(const Zero&, const BlockScaling&, Scalar,
                             const ConstVectorRef&, const ConstVectorRef&,
                             VectorRef) {
  schur_weight_is_materialised();
}

void apply_residual_scaling_impl(const Zero&, const BlockScaling&,
                                 const ConstVectorRef& v, VectorRef out) {
  out = v;
}

void complementarity_residual_impl(const Zero&, const BlockScaling&, Scalar,
                                   VectorRef) {
  degenerate_barrier();
}

void second_order_correction_impl(const Zero&, const BlockScaling&,
                                  const ConstVectorRef&, const ConstVectorRef&,
                                  VectorRef) {
  degenerate_barrier();
}

Scalar largest_feasible_step_impl(const Zero&, ConeSide, const ConstVectorRef&,
                                  const ConstVectorRef&) {
  no_interior();
}

void interior_direction_impl(const Zero&, ConeSide, VectorRef) {
  no_interior();
}

Scalar margin_impl(const Zero&, ConeSide, const ConstVectorRef&) {
  no_interior();
}

}
