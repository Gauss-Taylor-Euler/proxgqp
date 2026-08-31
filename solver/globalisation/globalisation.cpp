#include "globalisation/globalisation.hpp"

#include <cmath>
#include <limits>

#include "globalisation/merit.hpp"

#include <algorithm>

namespace proxgqp {
namespace {

struct UnitStep final : Globalisation {
  explicit UnitStep(const GlobalisationSettings& given) : settings(given) {}
  void setup(const Cones&, Index, Index) override {}
  StepPair step(const ProblemData&, const Cones&, const Iterate&,
                const ConstVectorRef&, const ConstVectorRef&,
                const ConstVectorRef&) override {
    return StepPair{settings.largest_step, settings.largest_step};
  }
  const char* name() const override { return "unit_step"; }
  GlobalisationSettings settings;
};

struct FractionToBoundary final : Globalisation {
  explicit FractionToBoundary(const GlobalisationSettings& given)
      : settings(given) {}

  void setup(const Cones&, Index, Index) override {}
  StepPair step(const ProblemData&, const Cones& cones, const Iterate& iterate,
                const ConstVectorRef&, const ConstVectorRef& ds,
                const ConstVectorRef& dz) override {
    if (offsets.size() != cones.size() + 1) offsets = block_offsets(cones);
    Scalar primal = settings.largest_step;
    Scalar dual = settings.largest_step;
    for (std::size_t j = 0; j < cones.size(); ++j) {
      if (!has_interior(cones[j])) continue;
      const Index start = offsets[j];
      const Index length = offsets[j + 1] - offsets[j];
      const Scalar fraction = is_curved(cones[j])
                                  ? settings.curved_boundary_fraction
                                  : settings.boundary_fraction;
      primal = std::min(
          primal, fraction * largest_feasible_step(
                                 cones[j], ConeSide::Primal,
                                 iterate.s.segment(start, length),
                                 ds.segment(start, length)));
      dual = std::min(
          dual, fraction * largest_feasible_step(
                               cones[j], ConeSide::Dual,
                               iterate.z.segment(start, length),
                               dz.segment(start, length)));
    }
    return StepPair{std::min(primal, settings.largest_step),
                    std::min(dual, settings.largest_step)};
  }

  const char* name() const override { return "fraction_to_boundary"; }
  GlobalisationSettings settings;
  std::vector<Index> offsets;
};

struct ExactSearch final : Globalisation {
  explicit ExactSearch(const GlobalisationSettings& given) : settings(given) {}

  void setup(const Cones& cones, Index columns, Index rows) override {
    merit.setup(cones, columns, rows);
  }

  StepPair step(const ProblemData& data, const Cones&, const Iterate& iterate,
                const ConstVectorRef& dx, const ConstVectorRef& ds,
                const ConstVectorRef&) override {
    merit.bind(data, iterate, dx, ds);
    const Scalar at_zero = merit.slope(0.0);
    if (!(at_zero < 0.0)) return unit();

    Scalar low = 0.0, slope_low = at_zero;
    Scalar high = settings.largest_step;
    Scalar slope_high = merit.slope(high);

    if (slope_high < 0.0) {
      const Scalar curvature = merit.curvature_lower_bound();
      low = high;
      slope_low = slope_high;
      high = (curvature > 0.0 && std::isfinite(curvature))
                 ? std::abs(at_zero) / curvature
                 : settings.largest_step;
      if (!(high > low)) return StepPair{low, low};
      slope_high = merit.slope(high);
    }

    if (slope_low <= 0.0 && slope_high >= 0.0) {
      const Scalar width_tolerance = std::sqrt(
          std::numeric_limits<Scalar>::epsilon());
      int retained = 0;
      for (std::size_t taken = 0; taken < settings.largest_secant; ++taken) {
        const Scalar width = high - low;
        if (!(width > width_tolerance * std::max(Scalar(1), high))) break;
        const Scalar denominator = slope_high - slope_low;
        Scalar trial = denominator > 0.0
                           ? low - slope_low * width / denominator
                           : 0.5 * (low + high);
        if (!(trial > low && trial < high)) trial = 0.5 * (low + high);
        const Scalar slope_trial = merit.slope(trial);
        if (slope_trial < 0.0) {
          if (retained < 0) slope_high *= 0.5;
          low = trial;
          slope_low = slope_trial;
          retained = -1;
        } else {
          if (retained > 0) slope_low *= 0.5;
          high = trial;
          slope_high = slope_trial;
          retained = 1;
        }
      }
    }
    if (!(high > 0.0) || !std::isfinite(high)) return unit();
    const Scalar taken = std::min(high, settings.largest_search_step);
    return StepPair{taken, taken};
  }

  const char* name() const override { return "exact_search"; }

 private:
  StepPair unit() const {
    return StepPair{settings.largest_step, settings.largest_step};
  }

  GlobalisationSettings settings;
  Merit merit;
};

}

std::unique_ptr<Globalisation> make_unit_step(
    const GlobalisationSettings& settings) {
  return std::make_unique<UnitStep>(settings);
}

std::unique_ptr<Globalisation> make_exact_search(
    const GlobalisationSettings& settings) {
  return std::make_unique<ExactSearch>(settings);
}


struct BacktrackingSearch final : Globalisation {
  BacktrackingSearch(const GlobalisationSettings& given, bool armijo)
      : settings(given), uses_slope(armijo) {}

  void setup(const Cones& cones, Index columns, Index rows) override {
    merit.setup(cones, columns, rows);
  }

  StepPair step(const ProblemData& data, const Cones&, const Iterate& iterate,
                const ConstVectorRef& dx, const ConstVectorRef& ds,
                const ConstVectorRef&) override {
    merit.bind(data, iterate, dx, ds);
    const Scalar at_zero = merit.slope(0.0);
    if (uses_slope && !(at_zero < 0.0)) return StepPair{1.0, 1.0};

    Scalar taken = settings.largest_step;
    for (std::size_t back = 0; back < settings.largest_backtrack; ++back) {
      const Scalar change = merit.decrease(taken);
      const bool accepted =
          uses_slope
              ? change <= settings.armijo_slope_fraction * taken * at_zero
              : change < 0.0;
      if (accepted) return StepPair{taken, taken};
      taken *= settings.backtrack_factor;
    }
    return StepPair{taken, taken};
  }

  const char* name() const override {
    return uses_slope ? "armijo_search" : "decrease_search";
  }

 private:
  GlobalisationSettings settings;
  bool uses_slope = true;
  Merit merit;
};

std::unique_ptr<Globalisation> make_armijo_search(
    const GlobalisationSettings& settings) {
  return std::make_unique<BacktrackingSearch>(settings, true);
}

std::unique_ptr<Globalisation> make_decrease_search(
    const GlobalisationSettings& settings) {
  return std::make_unique<BacktrackingSearch>(settings, false);
}

std::unique_ptr<Globalisation> make_fraction_to_boundary(
    const GlobalisationSettings& settings) {
  return std::make_unique<FractionToBoundary>(settings);
}

std::unique_ptr<Globalisation> make_globalisation(
    GlobalisationKind kind, const GlobalisationSettings& settings) {
  switch (kind) {
    case GlobalisationKind::FractionToBoundary:
      return make_fraction_to_boundary(settings);
    case GlobalisationKind::ExactSearch:
      return make_exact_search(settings);
    case GlobalisationKind::ArmijoSearch:
      return make_armijo_search(settings);
    case GlobalisationKind::DecreaseSearch:
      return make_decrease_search(settings);
    default:
      return make_unit_step(settings);
  }
}

}
