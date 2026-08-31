#pragma once

#include <vector>

#include "cone/kernel.hpp"
#include "cone_types.hpp"
#include "iterate.hpp"
#include "types.hpp"

namespace proxgqp {

enum class Status {
  Solved,
  PrimalInfeasible,
  DualInfeasible,
  MaxIterations,
  NumericalFailure,
};

const char* status_name(Status status);

struct KktReport {
  Scalar primal_residual = 0.0;
  Scalar dual_residual = 0.0;
  Scalar dual_cone_violation = 0.0;
  Scalar complementarity = 0.0;
  Scalar objective = 0.0;
  bool converged = false;
};

struct TerminationSettings {
  Scalar eps_absolute = 1e-9;
  Scalar eps_relative = 1e-9;
  Scalar eps_primal_infeasible = 1e-9;
  Scalar eps_dual_infeasible = 1e-9;
};

struct Termination {
  void setup(const Cones& cones, Index columns, Index rows);
  KktReport evaluate(const ProblemData& data, const Cones& cones,
                     const Vector& x, const Vector& z,
                     const TerminationSettings& settings);
  bool primal_infeasible(const ProblemData& data, const Cones& cones,
                         const Vector& dual_step,
                         const TerminationSettings& settings);
  bool dual_infeasible(const ProblemData& data, const Cones& cones,
                       const Vector& primal_step,
                       const TerminationSettings& settings);
  const Vector& slack() const { return recomputed_slack; }

 private:
  std::vector<Index> offsets;
  Vector recomputed_slack, projected, stationarity, work_rows, work_columns;
};

}
