#pragma once

#include <memory>
#include <vector>

#include "cone/kernel.hpp"
#include "cone_types.hpp"
#include "iterate.hpp"
#include "types.hpp"

namespace proxgqp {

enum class ScheduleKind { Bcl, Gbcl, Interior };
enum class Penalty { Bcl, GBcl };

struct ScheduleSettings {
  Scalar penalty_reduction = 0.05;
  Scalar smallest_penalty = 1e-13;
  Scalar smallest_equality_penalty = 1e-13;
  Scalar tighten_exponent = 0.9;
  Scalar loosen_exponent = 0.1;
  Scalar smallest_rho = 1e-13;
  Scalar smallest_proximal_slack = 1e-13;
  Scalar smallest_tolerance = 1e-13;
  Scalar proximal_slack_reduction = 0.1;
  bool schedule_rho = true;
  Scalar eps_regularisation = 1e-9;
  Scalar largest_reduction = 1.0;
  Scalar refactor_ratchet = 1.0;
  Scalar coarse_floor = 1e-10;
  Scalar fine_floor = 1e-13;
  Scalar improvement = 0.95;
  Scalar infeasibility_threshold = 0.9;
  Scalar stalled_rate = 0.666;
  std::size_t stalls_before_fine = 7;
  std::size_t early_outers = 5;
  Scalar initial_tolerance = 1e-2;
  Scalar rho_reduction = 0.1;
  Scalar violation_exponent = 1.0;
  Scalar smallest_factor = 1e-3;
  Scalar newton_tolerance = 1e-1;
  Scalar restart_penalty = 1.0 / 1.1;
  Scalar restart_threshold = 1e-5;
  Scalar mu_adapt = 2.0;
  Scalar initial_penalty = 1e-1;
  Scalar equality_penalty_cap = 1e-3;
  Scalar eps_absolute = 1e-9;
  std::size_t safe_guard = 30;
  bool revert_on_reject = false;
};

struct Schedule {
  virtual ~Schedule() = default;
  virtual void setup(const Cones& cones, const Iterate& iterate,
                     Scalar reference) = 0;
  virtual void update(Iterate& iterate, const Vector& violation,
                      Scalar reference, Scalar primal_residual,
                      Scalar dual_residual, bool inner_converged,
                      std::size_t inner_taken) = 0;
  virtual const Vector& tolerance() const = 0;
  virtual Scalar inner_tolerance() const = 0;
  virtual const char* name() const = 0;
};

std::unique_ptr<Schedule> make_schedule(ScheduleKind kind,
                                        const ScheduleSettings& settings);
std::unique_ptr<Schedule> make_bcl(const ScheduleSettings& settings);
std::unique_ptr<Schedule> make_gbcl(const ScheduleSettings& settings);
std::unique_ptr<Schedule> make_interior(const ScheduleSettings& settings);


}
