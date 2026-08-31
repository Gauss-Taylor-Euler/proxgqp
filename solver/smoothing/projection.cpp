#include <stdexcept>

#include "smoothing/smoothing.hpp"

namespace proxgqp {
namespace {

struct Projection final : Smoothing {
  explicit Projection(const SmoothingSettings& given) : settings(given) {}

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
    reference_point.setZero(rows);
    projected.setZero(rows);
    jacobian_image.setZero(rows);
    scaled_cone.setZero(rows);
    combined_slack.setZero(rows);
    constraint_image.setZero(rows);
    penalty.setZero(rows);
  }

  void residuals(const Iterate& iterate, const ProblemData& data) override {
    penalty = iterate.cone_penalty;
    reference_point =
        penalty.cwiseProduct(iterate.z_centre) - iterate.s;

    held.stationary_x.noalias() = *data.P * iterate.x;
    held.stationary_x += *data.q + iterate.rho * (iterate.x - iterate.x_centre);
    held.stationary_x.noalias() += data.E->transpose() * iterate.y;

    constraint_image.noalias() = *data.E * iterate.x;
    held.equality =
        iterate.equality_penalty.cwiseProduct(iterate.y - iterate.y_centre) -
        (constraint_image + iterate.s - *data.b);

    for (std::size_t j = 0; j < cones->size(); ++j) {
      const Index start = offsets[j];
      const Index length = offsets[j + 1] - offsets[j];
      const auto sigma = reference_point.segment(start, length);
      evaluate_projection((*cones)[j], sigma, projected.segment(start, length));
      build_projection_scaling((*cones)[j], sigma,
                               penalty.segment(start, length),
                               held_scalings[j]);
      apply_residual_scaling((*cones)[j], held_scalings[j],
                             iterate.z.segment(start, length),
                             jacobian_image.segment(start, length));
      materialise_operator((*cones)[j], held_scalings[j],
                           iterate.proximal_slack, held_blocks[j]);
    }
    held.cone = penalty.cwiseProduct(iterate.z) - projected;

    for (std::size_t j = 0; j < cones->size(); ++j) {
      const Index start = offsets[j];
      const Index length = offsets[j + 1] - offsets[j];
      apply_residual_scaling((*cones)[j], held_scalings[j],
                             held.cone.segment(start, length),
                             scaled_cone.segment(start, length));
    }
    scaled_cone = scaled_cone.cwiseQuotient(penalty);
    held.stationary_slack =
        iterate.proximal_slack * (iterate.s - iterate.s_centre) + iterate.y -
        jacobian_image;
    combined_slack = held.stationary_slack + scaled_cone;
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
      apply_residual_scaling((*cones)[j], held_scalings[j],
                             ds.segment(start, length),
                             jacobian_image.segment(start, length));
    }
    dz = -(jacobian_image + held.cone).cwiseQuotient(penalty);
  }

  bool corrector(const Iterate&, const ConstVectorRef&, const ConstVectorRef&,
                 Scalar, Scalar) override {
    return false;
  }

  Scalar centring() const override { return 0.0; }

  Scalar reference(const Iterate&) const override { return held.largest(); }

  bool requires_interior() const override { return false; }
  bool one_step_per_outer() const override { return false; }
  const char* name() const override { return "projection"; }

 private:
  SmoothingSettings settings;
  const Cones* cones = nullptr;
  std::vector<Index> offsets;
  SmoothingResiduals held;
  std::vector<GBlock> held_blocks;
  std::vector<BlockScaling> held_scalings;
  Vector reference_point, projected, jacobian_image, scaled_cone;
  Vector constraint_image, penalty, combined_slack;
};

}

std::unique_ptr<Smoothing> make_projection(const SmoothingSettings& settings) {
  return std::make_unique<Projection>(settings);
}

}
