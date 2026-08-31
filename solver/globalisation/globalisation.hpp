#pragma once

#include <memory>

#include "cone/kernel.hpp"
#include "cone_types.hpp"
#include "iterate.hpp"
#include "types.hpp"

namespace proxgqp {

struct StepPair {
  Scalar primal = 1.0;
  Scalar dual = 1.0;
};

enum class GlobalisationKind { UnitStep, FractionToBoundary, ExactSearch };
enum class LineSearchKind { Exact, Armijo, Decrease };

struct GlobalisationSettings {
  Scalar boundary_fraction = 0.995;
  Scalar curved_boundary_fraction = 0.8;
  Scalar largest_step = 1.0;
  std::size_t largest_secant = 40;
  Scalar largest_search_step = 1e6;
};

struct Globalisation {
  virtual ~Globalisation() = default;
  virtual void setup(const Cones& cones, Index columns, Index rows) = 0;
  virtual StepPair step(const ProblemData& data, const Cones& cones,
                        const Iterate& iterate, const ConstVectorRef& dx,
                        const ConstVectorRef& ds,
                        const ConstVectorRef& dz) = 0;
  virtual const char* name() const = 0;
};

std::unique_ptr<Globalisation> make_globalisation(
    GlobalisationKind kind, const GlobalisationSettings& settings);
std::unique_ptr<Globalisation> make_unit_step(
    const GlobalisationSettings& settings);
std::unique_ptr<Globalisation> make_fraction_to_boundary(
    const GlobalisationSettings& settings);
std::unique_ptr<Globalisation> make_exact_search(
    const GlobalisationSettings& settings);

}
