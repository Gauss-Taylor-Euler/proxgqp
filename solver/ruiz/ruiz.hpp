#pragma once

#include <vector>

#include "cone_types.hpp"
#include "iterate.hpp"
#include "types.hpp"

namespace proxgqp {

struct RuizSettings {
  std::size_t passes = 10;
  Scalar tolerance = 1e-3;
  bool scale_cost = true;
};

struct ScaledProblem {
  SparseMatrix P, E;
  Vector q, b;
  Vector column_scale;
  Vector row_scale;
  Scalar cost_scale = 1.0;
};

void equilibrate(const SparseMatrix& P, const Vector& q, const SparseMatrix& E,
                 const Vector& b, const Cones& cones,
                 const RuizSettings& settings, ScaledProblem& scaled);

void unscale_primal(const ScaledProblem& scaled, const Vector& x_scaled,
                    Vector& x);
void unscale_slack(const ScaledProblem& scaled, const Vector& s_scaled,
                   Vector& s);
void unscale_dual(const ScaledProblem& scaled, const Vector& z_scaled,
                  Vector& z);
void scale_primal(const ScaledProblem& scaled, const Vector& x, Vector& out);
void scale_slack(const ScaledProblem& scaled, const Vector& s, Vector& out);
void scale_dual(const ScaledProblem& scaled, const Vector& z, Vector& out);

}
