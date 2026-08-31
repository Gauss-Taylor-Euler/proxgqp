#include "initialisation/initialisation.hpp"

#include <algorithm>
#include <limits>

namespace proxgqp {
namespace detail {

void shift_inside(const Cones& cones, const std::vector<Index>& offsets,
                  ConeSide side, Scalar pad_constant, Scalar pad_fraction,
                  Vector& point) {
  Scalar smallest = std::numeric_limits<Scalar>::infinity();
  Scalar scale = 0.0;
  bool any = false;
  for (std::size_t j = 0; j < cones.size(); ++j) {
    if (!has_interior(cones[j])) continue;
    any = true;
    const Index start = offsets[j];
    const Index length = offsets[j + 1] - offsets[j];
    smallest = std::min(smallest,
                        margin(cones[j], side, point.segment(start, length)));
    scale = std::max(
        scale, point.segment(start, length).lpNorm<Eigen::Infinity>());
  }
  if (!any) return;
  const Scalar target = pad_constant + pad_fraction * scale;
  const Scalar shift = std::max(Scalar(0), target - smallest);
  if (shift == 0.0) return;
  Vector direction;
  for (std::size_t j = 0; j < cones.size(); ++j) {
    if (!has_interior(cones[j])) continue;
    const Index start = offsets[j];
    const Index length = offsets[j + 1] - offsets[j];
    direction.resize(length);
    interior_direction(cones[j], side, direction);
    point.segment(start, length) += shift * direction;
  }
}

void anchor_centres(Iterate& iterate) {
  iterate.x_centre = iterate.x;
  iterate.s_centre = iterate.s;
  iterate.z_centre = iterate.z;
  iterate.y_centre = iterate.y;
}

}

std::unique_ptr<Initialisation> make_initialisation(
    bool requires_interior, const InitialisationSettings& settings) {
  return requires_interior ? make_interior_start(settings)
                           : make_feasible_start(settings);
}

}
