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
  Scalar curvature_lower_bound() const { return curvature; }

 private:
  const Cones* cones = nullptr;
  const ProblemData* data = nullptr;
  const Iterate* iterate = nullptr;
  std::vector<Index> offsets;
  Vector direction_x, direction_slack;
  Vector trial_x, trial_slack, residual, multiplier, reference_point,
      projected, gradient_x, gradient_slack, work_rows;
  Scalar curvature = 0.0;
};

}
