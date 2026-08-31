#include "initialisation/initialisation.hpp"

namespace proxgqp {
namespace {

struct FeasibleStart final : Initialisation {
  explicit FeasibleStart(const InitialisationSettings& given)
      : settings(given) {}

  void cold(const ProblemData& data, const Cones& cones, Road&,
            Iterate& iterate) override {
    const std::vector<Index> offsets = block_offsets(cones);
    iterate.x.setZero();
    iterate.y.setZero();
    iterate.z.setZero();
    reflected = -*data.f;
    evaluate_projection_all(cones, offsets, reflected, reflected);
    iterate.s = *data.f + reflected;
    detail::anchor_centres(iterate);
  }

  void warm(const Cones&, Iterate& iterate) override {
    detail::anchor_centres(iterate);
  }

  const char* name() const override { return "feasible_start"; }

 private:
  InitialisationSettings settings;
  Vector reflected;
};

}

std::unique_ptr<Initialisation> make_feasible_start(
    const InitialisationSettings& settings) {
  return std::make_unique<FeasibleStart>(settings);
}

}
