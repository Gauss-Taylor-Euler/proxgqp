#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "smoothing/smoothing.hpp"

namespace proxgqp {
namespace {

struct Barrier final : Smoothing {
  explicit Barrier(const SmoothingSettings& given) : settings(given) {}

  void setup(const Cones& given_cones, Index columns, Index rows) override {
    cones = &given_cones;
    offsets = block_offsets(given_cones);
    if (offsets.back() != rows)
      throw std::invalid_argument(
          "the cone list's total dimension does not match the constraint rows");
    held.stationary_x.setZero(columns);
    held.stationary_slack.setZero(rows);
    held.cone.setZero(rows);
    held.equality.setZero(rows);
    held_blocks.assign(given_cones.size(), GBlock());
    held_scalings.assign(given_cones.size(), BlockScaling());
    scaled_cone.setZero(rows);
    combined_slack.setZero(rows);
    correction.setZero(rows);
    constraint_image.setZero(rows);
    barrier_degree = 0;
    for (std::size_t j = 0; j < given_cones.size(); ++j)
      if (has_interior(given_cones[j]))
        barrier_degree += degree(given_cones[j]);
    barred.assign(given_cones.size(), false);
    for (std::size_t j = 0; j < given_cones.size(); ++j)
      barred[j] = has_interior(given_cones[j]);
  }

  void residuals(const Iterate& iterate, const ProblemData& data) override {
    held.stationary_x.noalias() = *data.P * iterate.x;
    held.stationary_x += *data.q + iterate.rho * (iterate.x - iterate.x_centre);
    held.stationary_x.noalias() += data.E->transpose() * iterate.y;

    constraint_image.noalias() = *data.E * iterate.x;
    held.equality =
        iterate.equality_penalty.cwiseProduct(iterate.y - iterate.y_centre) -
        (constraint_image + iterate.s - *data.f);

    held.stationary_slack =
        iterate.proximal_slack * (iterate.s - iterate.s_centre) + iterate.y -
        iterate.z;

    measure = duality_measure(iterate);
    const Scalar target =
        settings.mehrotra ? 0.0 : settings.centring_fraction * measure;
    for (std::size_t j = 0; j < cones->size(); ++j) {
      const Index start = offsets[j];
      const Index length = offsets[j + 1] - offsets[j];
      const auto slack = iterate.s.segment(start, length);
      const auto dual = iterate.z.segment(start, length);
      if (barred[j]) {
        build_barrier_scaling((*cones)[j], slack, dual, held_scalings[j]);
        complementarity_residual((*cones)[j], held_scalings[j], target,
                                 held.cone.segment(start, length));
        apply_residual_scaling((*cones)[j], held_scalings[j],
                               held.cone.segment(start, length),
                               scaled_cone.segment(start, length));
      } else {
        const auto penalty = iterate.cone_penalty.segment(start, length);
        const auto centre = iterate.z_centre.segment(start, length);
        const Vector reference_point = penalty.cwiseProduct(centre) - slack;
        build_projection_scaling((*cones)[j], reference_point, penalty,
                                 held_scalings[j]);
        held.cone.segment(start, length) =
            penalty.cwiseProduct(dual - centre) + slack;
        scaled_cone.segment(start, length) =
            held.cone.segment(start, length).cwiseQuotient(penalty);
      }
      materialise_operator((*cones)[j], held_scalings[j],
                           iterate.proximal_slack, held_blocks[j]);
    }
    combined_slack = held.stationary_slack + scaled_cone;
    proximal_slack = iterate.proximal_slack;
  }

  const SmoothingResiduals& current() const override { return held; }
  const std::vector<GBlock>& blocks() const override { return held_blocks; }
  const std::vector<BlockScaling>& scalings() const override {
    return held_scalings;
  }

  Residuals road_residuals() const override {
    return Residuals{held.stationary_x, combined_slack, held.equality};
  }

  void recover_dual_step(const ConstVectorRef& ds, VectorRef dz) override {
    for (std::size_t j = 0; j < cones->size(); ++j) {
      const Index start = offsets[j];
      const Index length = offsets[j + 1] - offsets[j];
      Vector image(length);
      apply_block(held_blocks[j], ds.segment(start, length), image);
      image -= proximal_slack * ds.segment(start, length);
      dz.segment(start, length) = -image - scaled_cone.segment(start, length);
    }
  }

  bool corrector(const Iterate& iterate, const ConstVectorRef& ds,
                 const ConstVectorRef& dz, Scalar primal_step,
                 Scalar dual_step) override {
    if (!settings.mehrotra || measure <= 0.0 || barrier_degree == 0)
      return false;

    Scalar reached = 0.0;
    for (std::size_t j = 0; j < cones->size(); ++j) {
      if (!barred[j]) continue;
      const Index start = offsets[j];
      const Index length = offsets[j + 1] - offsets[j];
      reached +=
          (iterate.s.segment(start, length) +
           primal_step * ds.segment(start, length))
              .dot(iterate.z.segment(start, length) +
                   dual_step * dz.segment(start, length));
    }
    const Scalar ratio = std::min(
        1.0, std::max(0.0, (reached / static_cast<Scalar>(barrier_degree)) /
                               measure));
    centring_used = std::min(settings.largest_centring,
                             std::max(settings.smallest_centring,
                                      ratio * ratio * ratio));
    const Scalar target = centring_used * measure;

    for (std::size_t j = 0; j < cones->size(); ++j) {
      if (!barred[j]) continue;
      const Index start = offsets[j];
      const Index length = offsets[j + 1] - offsets[j];
      complementarity_residual((*cones)[j], held_scalings[j], target,
                               held.cone.segment(start, length));
      second_order_correction((*cones)[j], held_scalings[j],
                              ds.segment(start, length),
                              dz.segment(start, length),
                              correction.segment(start, length));
      held.cone.segment(start, length) += correction.segment(start, length);
      apply_residual_scaling((*cones)[j], held_scalings[j],
                             held.cone.segment(start, length),
                             scaled_cone.segment(start, length));
    }
    combined_slack = held.stationary_slack + scaled_cone;
    return true;
  }

  Scalar centring() const override { return centring_used; }

  Scalar reference(const Iterate& iterate) const override {
    return duality_measure(iterate);
  }

  bool requires_interior() const override { return true; }
  bool one_step_per_outer() const override { return true; }
  const char* name() const override { return "barrier"; }

 private:
  Scalar duality_measure(const Iterate& iterate) const {
    Scalar paired = 0.0;
    std::size_t total_degree = 0;
    for (std::size_t j = 0; j < cones->size(); ++j) {
      if (!barred[j]) continue;
      const Index start = offsets[j];
      const Index length = offsets[j + 1] - offsets[j];
      paired += iterate.s.segment(start, length)
                    .dot(iterate.z.segment(start, length));
      total_degree += degree((*cones)[j]);
    }
    return total_degree ? paired / static_cast<Scalar>(total_degree) : 0.0;
  }

  SmoothingSettings settings;
  const Cones* cones = nullptr;
  std::vector<Index> offsets;
  std::vector<bool> barred;
  SmoothingResiduals held;
  std::vector<GBlock> held_blocks;
  std::vector<BlockScaling> held_scalings;
  Vector scaled_cone, combined_slack, correction, constraint_image;
  std::size_t barrier_degree = 0;
  Scalar centring_used = 0.0;
  Scalar measure = 0.0;
  Scalar proximal_slack = 0.0;
};

}

std::unique_ptr<Smoothing> make_barrier(const SmoothingSettings& settings) {
  return std::make_unique<Barrier>(settings);
}

}
