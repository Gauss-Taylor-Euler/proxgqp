#include "smoothing/smoothing.hpp"

#include <algorithm>

namespace proxgqp {

Scalar SmoothingResiduals::largest() const {
  const auto largest_of = [](const Vector& block) {
    return block.size() ? block.lpNorm<Eigen::Infinity>() : Scalar(0);
  };
  return std::max({largest_of(stationary_x), largest_of(stationary_slack),
                   largest_of(cone), largest_of(equality)});
}

std::unique_ptr<Smoothing> make_smoothing(SmoothingKind kind,
                                          const SmoothingSettings& settings) {
  switch (kind) {
    case SmoothingKind::Barrier: return make_barrier(settings);
    default: return make_projection(settings);
  }
}

}
