#pragma once

#include <cstddef>
#include <vector>

#include "cone_types.hpp"
#include "gblock.hpp"
#include "types.hpp"

namespace proxgqp {

enum class ConeSide { Primal, Dual };

enum class ScalingSource { Projection, Barrier };

enum class SchurAssembly { Materialise, ByApplication };

struct BlockScaling {
  ScalingSource source = ScalingSource::Barrier;
  Vector lambda;
  Vector scaling_point;
  Vector barrier_gradient;
  Vector dual_point;
  Vector linearisation_point;
  Vector penalty;
  Vector congruence_eigenvalues;
  DenseMatrix congruence_root;
  DenseMatrix inverse_congruence_root;
  DenseMatrix projection_jacobian;
  DenseMatrix eigenvectors;
  DenseMatrix divided_differences;
  DenseMatrix congruence_eigenvectors;
  DenseMatrix barrier_hessian;
  Scalar scaling_factor = 1.0;
  Scalar complementarity = 0.0;
};

Index rows(const Cone& cone);
std::size_t degree(const Cone& cone);
bool is_curved(const Cone& cone);
bool has_interior(const Cone& cone);
bool contains(const Cone& cone, ConeSide side, const ConstVectorRef& v);

void evaluate_projection(const Cone& cone, const ConstVectorRef& v,
                         VectorRef out);
void build_projection_scaling(const Cone& cone, const ConstVectorRef& sigma,
                              const ConstVectorRef& mu, BlockScaling& scaling);
void build_barrier_scaling(const Cone& cone, const ConstVectorRef& s,
                           const ConstVectorRef& z, BlockScaling& scaling);
void materialise_operator(const Cone& cone, const BlockScaling& scaling,
                          Scalar rho_p, GBlock& block);
SchurAssembly schur_assembly(const Cone& cone);
GBlock::Kind widest_operator_kind(const Cone& cone);
void apply_schur_weight(const Cone& cone, const BlockScaling& scaling,
                        Scalar rho_p, const ConstVectorRef& mu_s,
                        const ConstVectorRef& v, VectorRef out);
void apply_residual_scaling(const Cone& cone, const BlockScaling& scaling,
                            const ConstVectorRef& v, VectorRef out);
void complementarity_residual(const Cone& cone, const BlockScaling& scaling,
                              Scalar target, VectorRef out);
void second_order_correction(const Cone& cone, const BlockScaling& scaling,
                             const ConstVectorRef& ds, const ConstVectorRef& dz,
                             VectorRef out);
Scalar largest_feasible_step(const Cone& cone, ConeSide side,
                             const ConstVectorRef& v, const ConstVectorRef& dv);
void interior_direction(const Cone& cone, ConeSide side, VectorRef out);
Scalar margin(const Cone& cone, ConeSide side, const ConstVectorRef& v);

std::vector<Index> block_offsets(const Cones& cones);

void evaluate_projection_all(const Cones& cones,
                             const std::vector<Index>& offsets,
                             const Vector& v, Vector& out);

namespace detail {

[[noreturn]] void schur_weight_is_materialised();

#define PROXGQP_DECLARE_KERNEL(CONE)                                          \
  Index rows_impl(const CONE& cone);                                          \
  std::size_t degree_impl(const CONE& cone);                                  \
  bool is_curved_impl(const CONE& cone);                                      \
  bool has_interior_impl(const CONE& cone);                                    \
  bool contains_impl(const CONE& cone, ConeSide side,                         \
                     const ConstVectorRef& v);                                \
  void evaluate_projection_impl(const CONE& cone, const ConstVectorRef& v,    \
                                VectorRef out);                               \
  void build_projection_scaling_impl(                                         \
      const CONE& cone, const ConstVectorRef& sigma,                          \
      const ConstVectorRef& mu, BlockScaling& scaling);                       \
  void build_barrier_scaling_impl(const CONE& cone, const ConstVectorRef& s,  \
                                  const ConstVectorRef& z,                    \
                                  BlockScaling& scaling);                     \
  void materialise_operator_impl(const CONE& cone,                            \
                                 const BlockScaling& scaling, Scalar rho_p,   \
                                 GBlock& block);                              \
  SchurAssembly schur_assembly_impl(const CONE& cone);                        \
  GBlock::Kind widest_operator_kind_impl(const CONE& cone);              \
  void apply_schur_weight_impl(                                               \
      const CONE& cone, const BlockScaling& scaling, Scalar rho_p,            \
      const ConstVectorRef& mu_s, const ConstVectorRef& v, VectorRef out);    \
  void apply_residual_scaling_impl(const CONE& cone,                          \
                                   const BlockScaling& scaling,               \
                                   const ConstVectorRef& v, VectorRef out);   \
  void complementarity_residual_impl(const CONE& cone,                        \
                                     const BlockScaling& scaling,             \
                                     Scalar target, VectorRef out);           \
  void second_order_correction_impl(                                          \
      const CONE& cone, const BlockScaling& scaling, const ConstVectorRef& ds,\
      const ConstVectorRef& dz, VectorRef out);                               \
  Scalar largest_feasible_step_impl(const CONE& cone, ConeSide side,          \
                                    const ConstVectorRef& v,                  \
                                    const ConstVectorRef& dv);                \
  void interior_direction_impl(const CONE& cone, ConeSide side,               \
                               VectorRef out);                                \
  Scalar margin_impl(const CONE& cone, ConeSide side, const ConstVectorRef& v);

PROXGQP_DECLARE_KERNEL(Zero)
PROXGQP_DECLARE_KERNEL(Nonneg)
PROXGQP_DECLARE_KERNEL(SecondOrder)
PROXGQP_DECLARE_KERNEL(PSDTriangle)
PROXGQP_DECLARE_KERNEL(Exponential)
PROXGQP_DECLARE_KERNEL(Power)

#undef PROXGQP_DECLARE_KERNEL

}
}
