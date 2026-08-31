#pragma once

#include "cone_types.hpp"
#include "iterate.hpp"
#include <vector>

#include "types.hpp"

namespace proxgqp {

struct Merit {
  void setup(const Cones& cones, Index columns, Index rows);
  void bind(const ProblemData& data, const Iterate& iterate,
            const ConstVectorRef& dx, const ConstVectorRef& ds);
  Scalar slope(Scalar step);
  Scalar decrease(Scalar step);
  Scalar curvature_lower_bound() const { return curvature; }

 private:
  const Cones* cones = nullptr;
  std::vector<Index> offsets;
  Vector direction_slack, direction_image, residual, multiplier;
  Vector reference_base, reference_point, projected, slack_over_penalty;
  Vector objective_image;
  Vector penalty;
  Scalar constant = 0.0;
  Scalar curvature = 0.0;
  Scalar base_energy = 0.0;
};

}
