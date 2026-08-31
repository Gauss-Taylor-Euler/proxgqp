#pragma once

#include <cstddef>

#include "termination/termination.hpp"
#include "types.hpp"

namespace proxgqp {

struct Results {
  Vector x, s, z;
  Status status = Status::MaxIterations;
  KktReport kkt;
  std::size_t outer_iterations = 0;
  std::size_t inner_iterations = 0;
  Scalar seconds = 0.0;
  bool has_point = false;
};

}
