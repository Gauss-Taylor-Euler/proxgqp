#pragma once

#include <cstddef>

#include "globalisation/globalisation.hpp"
#include "initialisation/initialisation.hpp"
#include "road/road.hpp"
#include "ruiz/ruiz.hpp"
#include "schedule/schedule.hpp"
#include "smoothing/smoothing.hpp"
#include "termination/termination.hpp"
#include "types.hpp"

namespace proxgqp {

enum class Method { Semismooth, Interior, Auto };

struct Shared {
  Scalar rho = 1e-6;
  Scalar rho_p = 1e-4;
  Scalar mu_eq = 1e-3;
  Scalar rho_p_decay = 1.0;
  Scalar rho_p_min = 1e-12;
  Scalar refactor_bump = 10.0;
  std::size_t max_refactor = 10;
  std::size_t ruiz_iter = 10;
  std::size_t refine = 8;
  std::size_t infeas_repeat = 2;
  Scalar eps_infeas = 1e-9;
  Scalar stall_tol = 1e-16;
};

struct Tuning {

  struct Bcl {
    Scalar mu_in = 1e-1;
    Scalar mu_eq = 1e-3;
    Scalar alpha = 0.1;
    Scalar beta = 0.9;
    Scalar mu_update_factor = 0.1;
    Scalar mu_min_in = 1e-8;
    Scalar mu_min_eq = 1e-9;
    Scalar cold_reset = 1.0 / 1.1;
    Scalar cold_reset_threshold = 1e-5;
    Scalar rho_p_outer = 0.1;
    std::size_t safe_guard = 10000;
    bool revert_on_reject = true;
  } bcl;

  struct Gbcl {
    Scalar mu_in = 1e-1;
    Scalar mu_eq = 1e-3;
    Scalar alpha = 0.1;
    Scalar beta = 0.9;
    Scalar mu_exp = 2.0;
    Scalar mu_adapt = 2.0;
    Scalar mu_min = 1e-9;
    Scalar penalty_reduction = 0.3;
    Scalar eps_outer_init = 1e-2;
    Scalar eps_newton_init = 1e-1;
    Scalar rho_p_outer = 0.1;
    std::size_t safe_guard = 30;
  } gbcl;

  struct Semismooth : Shared {
    Scalar mu_in = 1e-1;
    LineSearchKind line_search = LineSearchKind::Exact;
    Penalty penalty = Penalty::GBcl;
    Scalar ls_sigma = 1e-4;
    Scalar ls_beta = 0.5;
    std::size_t ls_max_back = 60;
    std::size_t ls_max_secant = 40;
  } semismooth;

  struct Interior : Shared {
    Interior() { infeas_repeat = 12; }
    Scalar tau = 0.99;
    Scalar tau_curved = 0.8;
    bool mehrotra = true;
    Scalar sigma_centre = 0.1;
    Scalar sigma_min = 1e-6;
    Scalar sigma_max = 1.0;
    Scalar eps_reg = 1e-9;
    bool seeded_start = true;
    Scalar pad_constant = 1.0;
    Scalar pad_fraction = 0.1;
    Scalar reg_floor = 1e-10;
    Scalar reg_fine = 1e-13;
    Scalar improvement = 0.95;
    Scalar infeasibility_threshold = 0.9;
    Scalar stalled_rate = 0.666;
    Scalar degenerate_step = 0.0;
    Scalar refactor_ratchet = 10.0;
    std::size_t stall_before_fine = 7;
    std::size_t early_outers = 5;
  } interior;


  const Shared& shared(Method method) const {
    if (method == Method::Interior) return interior;
    return semismooth;
  }
};

struct Settings {
  Method method = Method::Auto;

  Scalar eps_abs = 1e-9;
  Scalar eps_rel = 1e-9;
  Scalar eps_gap_abs = 1e-8;
  Scalar eps_gap_rel = 1e-8;

  std::size_t max_iter_outer = 1000;
  std::size_t max_iter_inner = 60;
  std::size_t max_newton = 4000;

  RoadKind road = RoadKind::ThreeByThree;
  BackendKind backend = BackendKind::Auto;
  bool equilibrate = true;
  bool scale_cost = false;
  std::size_t max_threads = 1;
  bool verbose = false;

  Tuning tuning;
};

inline Method resolve_method(Method requested) {
  if (requested != Method::Auto) return requested;
  return Method::Interior;
}

}
