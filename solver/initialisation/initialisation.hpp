#pragma once

#include <memory>
#include <vector>

#include "cone/kernel.hpp"
#include "cone_types.hpp"
#include "iterate.hpp"
#include "road/road.hpp"
#include "types.hpp"

namespace proxgqp {

struct InitialisationSettings {
  bool seeded = true;
  Scalar pad_constant = 1.0;
  Scalar pad_fraction = 0.1;
};

struct Initialisation {
  virtual ~Initialisation() = default;

  virtual void cold(const ProblemData& data, const Cones& cones, Road& road,
                    Iterate& iterate) = 0;
  virtual void warm(const Cones& cones, Iterate& iterate) = 0;
  virtual const char* name() const = 0;
};

std::unique_ptr<Initialisation> make_initialisation(
    bool requires_interior, const InitialisationSettings& settings);
std::unique_ptr<Initialisation> make_interior_start(
    const InitialisationSettings& settings);
std::unique_ptr<Initialisation> make_feasible_start(
    const InitialisationSettings& settings);

namespace detail {

void shift_inside(const Cones& cones, const std::vector<Index>& offsets,
                  ConeSide side, Scalar pad_constant, Scalar pad_fraction,
                  Vector& point);

void anchor_centres(Iterate& iterate);

void centre_complementarity(const Cones& cones,
                            const std::vector<Index>& offsets,
                            Vector& slack, Vector& dual);

}
}
