#pragma once

#include "cone_types.hpp"
#include "results.hpp"
#include "settings.hpp"
#include "types.hpp"

namespace proxgqp {

Results solve(const SparseMatrix& P, const Vector& q, const SparseMatrix& E,
              const Vector& f, const Cones& cones, const Settings& settings,
              const Results* warm = nullptr);

}
