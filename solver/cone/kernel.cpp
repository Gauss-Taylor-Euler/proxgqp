#include "cone/kernel.hpp"

#include <cassert>
#include <stdexcept>
#include <variant>

namespace proxgqp {
namespace detail {

void schur_weight_is_materialised() {
  throw std::domain_error(
      "this cone reports SchurAssembly::Materialise: its operator and Schur "
      "weight are formed as entries and inverted by structure, so there is no "
      "action to apply. Only a ByApplication cone answers these");
}

}

Index rows(const Cone& cone) {
  return std::visit([](const auto& block) { return detail::rows_impl(block); },
                    cone);
}

std::size_t degree(const Cone& cone) {
  return std::visit(
      [](const auto& block) { return detail::degree_impl(block); }, cone);
}

bool is_curved(const Cone& cone) {
  return std::visit(
      [](const auto& block) { return detail::is_curved_impl(block); }, cone);
}

bool has_interior(const Cone& cone) {
  return std::visit(
      [](const auto& block) { return detail::has_interior_impl(block); }, cone);
}

bool contains(const Cone& cone, ConeSide side, const ConstVectorRef& v) {
  return std::visit(
      [&](const auto& block) { return detail::contains_impl(block, side, v); },
      cone);
}

void evaluate_projection(const Cone& cone, const ConstVectorRef& v,
                         VectorRef out) {
  std::visit(
      [&](const auto& block) {
        detail::evaluate_projection_impl(block, v, out);
      },
      cone);
}

void build_projection_scaling(const Cone& cone, const ConstVectorRef& sigma,
                              const ConstVectorRef& mu, BlockScaling& scaling) {
  std::visit(
      [&](const auto& kind) {
        detail::build_projection_scaling_impl(kind, sigma, mu, scaling);
      },
      cone);
}

void build_barrier_scaling(const Cone& cone, const ConstVectorRef& s,
                           const ConstVectorRef& z, BlockScaling& scaling) {
  std::visit(
      [&](const auto& kind) {
        detail::build_barrier_scaling_impl(kind, s, z, scaling);
      },
      cone);
}

void materialise_operator(const Cone& cone, const BlockScaling& scaling,
                          Scalar rho_p, GBlock& block) {
  std::visit(
      [&](const auto& kind) {
        detail::materialise_operator_impl(kind, scaling, rho_p, block);
      },
      cone);
}

SchurAssembly schur_assembly(const Cone& cone) {
  return std::visit(
      [](const auto& block) { return detail::schur_assembly_impl(block); },
      cone);
}

void apply_schur_weight(const Cone& cone, const BlockScaling& scaling,
                        Scalar rho_p, const ConstVectorRef& mu_s,
                        const ConstVectorRef& v, VectorRef out) {
  std::visit(
      [&](const auto& kind) {
        detail::apply_schur_weight_impl(kind, scaling, rho_p, mu_s, v, out);
      },
      cone);
}

void apply_residual_scaling(const Cone& cone, const BlockScaling& scaling,
                            const ConstVectorRef& v, VectorRef out) {
  std::visit(
      [&](const auto& block) {
        detail::apply_residual_scaling_impl(block, scaling, v, out);
      },
      cone);
}

void complementarity_residual(const Cone& cone, const BlockScaling& scaling,
                              Scalar target, VectorRef out) {
  assert(scaling.source == ScalingSource::Barrier);
  std::visit(
      [&](const auto& block) {
        detail::complementarity_residual_impl(block, scaling, target, out);
      },
      cone);
}

void second_order_correction(const Cone& cone, const BlockScaling& scaling,
                             const ConstVectorRef& ds, const ConstVectorRef& dz,
                             VectorRef out) {
  assert(scaling.source == ScalingSource::Barrier);
  std::visit(
      [&](const auto& block) {
        detail::second_order_correction_impl(block, scaling, ds, dz, out);
      },
      cone);
}

Scalar largest_feasible_step(const Cone& cone, ConeSide side,
                             const ConstVectorRef& v,
                             const ConstVectorRef& dv) {
  return std::visit(
      [&](const auto& block) {
        return detail::largest_feasible_step_impl(block, side, v, dv);
      },
      cone);
}

void interior_direction(const Cone& cone, ConeSide side, VectorRef out) {
  std::visit(
      [&](const auto& block) {
        detail::interior_direction_impl(block, side, out);
      },
      cone);
}

Scalar margin(const Cone& cone, ConeSide side, const ConstVectorRef& v) {
  return std::visit(
      [&](const auto& block) { return detail::margin_impl(block, side, v); },
      cone);
}

GBlock::Kind widest_operator_kind(const Cone& cone) {
  return std::visit(
      [](const auto& block) {
        return detail::widest_operator_kind_impl(block);
      },
      cone);
}

std::vector<Index> block_offsets(const Cones& cones) {
  std::vector<Index> offsets(cones.size() + 1, 0);
  for (std::size_t block = 0; block < cones.size(); ++block)
    offsets[block + 1] = offsets[block] + rows(cones[block]);
  return offsets;
}

void evaluate_projection_all(const Cones& cones,
                             const std::vector<Index>& offsets, const Vector& v,
                             Vector& out) {
  for (std::size_t block = 0; block < cones.size(); ++block) {
    const Index start = offsets[block];
    const Index length = offsets[block + 1] - offsets[block];
    evaluate_projection(cones[block], v.segment(start, length),
                        out.segment(start, length));
  }
}


}
