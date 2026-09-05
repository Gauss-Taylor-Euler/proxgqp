#pragma once

#include <cmath>
#include <limits>

#include "types.hpp"

namespace proxgqp::detail {

using Vector3 = Eigen::Matrix<Scalar, 3, 1>;
using Matrix3 = Eigen::Matrix<Scalar, 3, 3>;

constexpr Scalar kSearchLimit = 1e12;
constexpr Scalar kUnboundedProbe = 1e6;
constexpr int kBisectionSteps = 60;

template <typename ConeType, typename Inside>
Scalar bisect_margin(const ConeType& cone, const ConstVectorRef& point,
                     const Vector3& interior_direction, Inside inside) {
  const auto inside_at = [&](Scalar shift) {
    return inside(cone, Vector3(point(0) - shift * interior_direction(0),
                                point(1) - shift * interior_direction(1),
                                point(2) - shift * interior_direction(2)));
  };
  Scalar lower = 0.0;
  Scalar upper = 0.0;
  if (inside_at(0.0)) {
    upper = 1.0;
    while (inside_at(upper) && upper < kSearchLimit) {
      lower = upper;
      upper *= 2.0;
    }
    if (upper >= kSearchLimit) return kSearchLimit;
  } else {
    lower = -1.0;
    while (!inside_at(lower) && lower > -kSearchLimit) {
      upper = lower;
      lower *= 2.0;
    }
    if (lower <= -kSearchLimit) return -kSearchLimit;
  }
  for (int step = 0; step < kBisectionSteps; ++step) {
    const Scalar middle = 0.5 * (lower + upper);
    (inside_at(middle) ? lower : upper) = middle;
  }
  return lower;
}

template <typename ConeType, typename Inside>
Scalar bisect_step(const ConeType& cone, const ConstVectorRef& point,
                   const ConstVectorRef& step_direction, Inside inside) {
  const Scalar unbounded = std::numeric_limits<Scalar>::infinity();
  const auto inside_at = [&](Scalar step) {
    return inside(cone,
                  Vector3(point(0) + step * step_direction(0),
                          point(1) + step * step_direction(1),
                          point(2) + step * step_direction(2)));
  };
  if (inside_at(1.0) && inside_at(kUnboundedProbe)) return unbounded;
  Scalar lower = 0.0;
  Scalar upper = 1.0;
  while (inside_at(upper) && upper < kSearchLimit) {
    lower = upper;
    upper *= 2.0;
  }
  if (upper >= kSearchLimit) return unbounded;
  for (int step = 0; step < kBisectionSteps; ++step) {
    const Scalar middle = 0.5 * (lower + upper);
    (inside_at(middle) ? lower : upper) = middle;
  }
  return lower;
}

}
