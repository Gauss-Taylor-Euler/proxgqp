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
  Scalar initial_tolerance = 0.1;
  Scalar tighten_exponent = 0.9;
  Scalar loosen_exponent = 0.1;
  Scalar rho_reduction = 0.1;
  Scalar smallest_rho = 1e-13;
  Scalar smallest_proximal_slack = 1e-13;
  Scalar smallest_tolerance = 1e-13;
  Scalar proximal_slack_reduction = 0.1;
  Scalar violation_exponent = 1.0;
  Scalar smallest_factor = 1e-3;
  bool schedule_rho = true;
  Scalar eps_regularisation = 1e-9;
  Scalar coarse_floor = 1e-10;
  Scalar fine_floor = 1e-13;
  Scalar improvement = 0.95;
  Scalar infeasibility_threshold = 0.9;
  Scalar stalled_rate = 0.666;
  std::size_t stalls_before_fine = 7;
  std::size_t early_outers = 5;
};

struct Schedule {
  virtual ~Schedule() = default;
  virtual void setup(const Cones& cones, const Iterate& iterate,
                     Scalar reference) = 0;
  virtual void update(Iterate& iterate, const Vector& violation,
                      Scalar reference, Scalar primal_residual,
                      Scalar dual_residual, bool inner_converged) = 0;
  virtual const Vector& tolerance() const = 0;
  virtual const char* name() const = 0;
};

std::unique_ptr<Schedule> make_schedule(ScheduleKind kind,
                                        const ScheduleSettings& settings);
std::unique_ptr<Schedule> make_bcl(const ScheduleSettings& settings);
std::unique_ptr<Schedule> make_gbcl(const ScheduleSettings& settings);
std::unique_ptr<Schedule> make_interior(const ScheduleSettings& settings);

void flatten_on_curved(const Cones& cones, const std::vector<Index>& offsets,
                       Vector& values);

}
