#include <algorithm>

#include "initialisation/initialisation.hpp"

namespace proxgqp {
namespace {

struct InteriorStart final : Initialisation {
  explicit InteriorStart(const InitialisationSettings& given)
      : settings(given) {}

  void cold(const ProblemData& data, const Cones& cones, Road& road,
            Iterate& iterate) override {
    const std::vector<Index> offsets = block_offsets(cones);
    reference_point(cones, offsets, iterate);
    if (settings.seeded &&
        solve_for_centre(data, cones, offsets, road, iterate)) {
      detail::shift_inside(cones, offsets, ConeSide::Primal,
                           settings.pad_constant, settings.pad_fraction,
                           iterate.s);
      detail::shift_inside(cones, offsets, ConeSide::Dual,
                           settings.pad_constant, settings.pad_fraction,
                           iterate.z);
    }
    detail::anchor_centres(iterate);
  }

  void warm(const Cones& cones, Iterate& iterate) override {
    const std::vector<Index> offsets = block_offsets(cones);
    detail::shift_inside(cones, offsets, ConeSide::Primal,
                         settings.pad_constant, settings.pad_fraction,
                         iterate.s);
    detail::shift_inside(cones, offsets, ConeSide::Dual, settings.pad_constant,
                         settings.pad_fraction, iterate.z);
    detail::anchor_centres(iterate);
  }

  const char* name() const override { return "interior_start"; }

 private:
  void reference_point(const Cones& cones, const std::vector<Index>& offsets,
                       Iterate& iterate) {
    iterate.x.setZero();
    iterate.y.setZero();
    Vector direction;
    for (std::size_t j = 0; j < cones.size(); ++j) {
      const Index start = offsets[j];
      const Index length = offsets[j + 1] - offsets[j];
      if (!has_interior(cones[j])) {
        iterate.s.segment(start, length).setZero();
        iterate.z.segment(start, length).setZero();
        continue;
      }
      direction.resize(length);
      interior_direction(cones[j], ConeSide::Primal, direction);
      iterate.s.segment(start, length) = direction;
      interior_direction(cones[j], ConeSide::Dual, direction);
      iterate.z.segment(start, length) = direction;
    }
  }

  bool solve_for_centre(const ProblemData& data, const Cones& cones,
                        const std::vector<Index>& offsets, Road& road,
                        Iterate& iterate) {
    blocks.assign(cones.size(), GBlock());
    scalings.assign(cones.size(), BlockScaling());
    for (std::size_t j = 0; j < cones.size(); ++j) {
      const Index start = offsets[j];
      const Index length = offsets[j + 1] - offsets[j];
      if (has_interior(cones[j])) {
        build_barrier_scaling(cones[j], iterate.s.segment(start, length),
                              iterate.z.segment(start, length), scalings[j]);
      } else {
        build_projection_scaling(
            cones[j], iterate.s.segment(start, length),
            iterate.cone_penalty.segment(start, length), scalings[j]);
      }
      materialise_operator(cones[j], scalings[j], iterate.proximal_slack,
                           blocks[j]);
    }
    road.values(*data.P, *data.E, iterate.rho);
    road.assemble(blocks, scalings, iterate.proximal_slack,
                  iterate.equality_penalty);
    if (!road.factor()) return false;

    const Index rows = iterate.s.size();
    empty_slack.setZero(rows);
    Residuals residuals{*data.q, empty_slack, *data.b};
    step_x.setZero(iterate.x.size());
    step_slack.setZero(rows);
    step_multiplier.setZero(rows);
    Direction direction{step_x, step_slack, step_multiplier};
    road.solve(residuals, direction);

    iterate.x = step_x;
    iterate.s = step_slack;
    iterate.y = step_multiplier;
    image.setZero(rows);
    for (std::size_t j = 0; j < cones.size(); ++j) {
      const Index start = offsets[j];
      const Index length = offsets[j + 1] - offsets[j];
      apply_block(blocks[j], step_slack.segment(start, length),
                  image.segment(start, length));
    }
    iterate.z = -(image - iterate.proximal_slack * step_slack);
    return true;
  }

  InitialisationSettings settings;
  std::vector<GBlock> blocks;
  std::vector<BlockScaling> scalings;
  Vector empty_slack, step_x, step_slack, step_multiplier, image;
};

}

std::unique_ptr<Initialisation> make_interior_start(
    const InitialisationSettings& settings) {
  return std::make_unique<InteriorStart>(settings);
}

}
