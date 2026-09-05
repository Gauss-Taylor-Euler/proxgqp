#include "solve.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>

#include "cone/kernel.hpp"

namespace proxgqp {
namespace {

using Clock = std::chrono::steady_clock;

}

Results solve(const SparseMatrix& P, const Vector& q, const SparseMatrix& E,
              const Vector& b, const Cones& cones, const Settings& settings,
              const Results* warm) {
  const auto started = Clock::now();
  Results results;
  const Index columns = P.rows();
  const Index rows = E.rows();
  const std::vector<Index> offsets = block_offsets(cones);
  validate(cones);

  const Tuning& tuning_early = settings.tuning;
  const Shared& shared_early = tuning_early.shared(settings.method);
  ScaledProblem scaled;
  if (settings.equilibrate) {
    RuizSettings ruiz_settings;
    ruiz_settings.passes = shared_early.ruiz_iter;
    ruiz_settings.scale_cost = settings.scale_cost;
    equilibrate(P, q, E, b, cones, ruiz_settings, scaled);
  } else {
    scaled.P = P; scaled.q = q; scaled.E = E; scaled.b = b;
    scaled.column_scale.setOnes(columns);
    scaled.row_scale.setOnes(rows);
    scaled.cost_scale = 1.0;
  }
  const ProblemData working{&scaled.P, &scaled.q, &scaled.E, &scaled.b};
  const ProblemData original{&P, &q, &E, &b};

  const Tuning& tuning = settings.tuning;
  const bool interior = settings.method != Method::Semismooth;
  const Tuning::Interior& barrier =
      settings.method == Method::InteriorExp ? tuning.interior_exp
                                             : tuning.interior;
  const Shared& shared = tuning.shared(settings.method);

  RoadSettings road_settings;
  road_settings.backend = settings.backend;
  road_settings.refinement_budget = shared.refine;
  road_settings.max_threads = settings.max_threads;
  auto road = make_road(
      resolve_road(settings.road, scaled.E.cols(), scaled.E.rows(),
                   Index(scaled.E.nonZeros())),
      road_settings);
  road->setup(scaled.P, scaled.E, cones);

  SmoothingSettings smoothing_settings;
  smoothing_settings.mehrotra = barrier.mehrotra;
  smoothing_settings.centring_fraction = barrier.sigma_centre;
  smoothing_settings.smallest_centring = barrier.sigma_min;
  smoothing_settings.largest_centring = barrier.sigma_max;
  auto smoothing = make_smoothing(
      interior ? SmoothingKind::Barrier : SmoothingKind::Projection,
      smoothing_settings);
  smoothing->setup(cones, columns, rows);

  GlobalisationSettings globalisation_settings;
  globalisation_settings.boundary_fraction = barrier.tau;
  globalisation_settings.curved_boundary_fraction = barrier.tau_curved;
  globalisation_settings.largest_secant = tuning.semismooth.ls_max_secant;
  globalisation_settings.armijo_slope_fraction = tuning.semismooth.ls_sigma;
  globalisation_settings.backtrack_factor = tuning.semismooth.ls_beta;
  globalisation_settings.largest_backtrack = tuning.semismooth.ls_max_back;
  GlobalisationKind globalisation_kind = GlobalisationKind::FractionToBoundary;
  if (!interior) {
    switch (tuning.semismooth.line_search) {
      case LineSearchKind::Armijo:
        globalisation_kind = GlobalisationKind::ArmijoSearch;
        break;
      case LineSearchKind::Decrease:
        globalisation_kind = GlobalisationKind::DecreaseSearch;
        break;
      default:
        globalisation_kind = GlobalisationKind::ExactSearch;
        break;
    }
  }
  auto globalisation = make_globalisation(globalisation_kind,
                                          globalisation_settings);
  globalisation->setup(cones, columns, rows);

  ScheduleSettings schedule_settings;
  schedule_settings.schedule_rho = interior;
  schedule_settings.eps_absolute = settings.eps_abs;
  const Penalty which = tuning.semismooth.penalty;
  Scalar initial_cone_penalty = tuning.semismooth.mu_in;
  Scalar initial_equality_penalty = shared.mu_eq;
  if (!interior) {
    if (which == Penalty::Bcl) {
      const auto& bcl = tuning.bcl;
      initial_cone_penalty = bcl.mu_in;
      initial_equality_penalty = bcl.mu_eq;
      schedule_settings.loosen_exponent = bcl.alpha;
      schedule_settings.tighten_exponent = bcl.beta;
      schedule_settings.penalty_reduction = bcl.mu_update_factor;
      schedule_settings.smallest_penalty = bcl.mu_min_in;
      schedule_settings.smallest_equality_penalty = bcl.mu_min_eq;
      schedule_settings.restart_penalty = bcl.cold_reset;
      schedule_settings.restart_threshold = bcl.cold_reset_threshold;
      schedule_settings.proximal_slack_reduction = bcl.rho_p_outer;
      schedule_settings.safe_guard = bcl.safe_guard;
      schedule_settings.revert_on_reject = bcl.revert_on_reject;
    } else if (which == Penalty::GBclExp) {
      const auto& gbcl = tuning.gbcl_exp;
      initial_cone_penalty = gbcl.mu_in;
      initial_equality_penalty = gbcl.mu_eq;
      schedule_settings.loosen_exponent = gbcl.alpha;
      schedule_settings.tighten_exponent = gbcl.beta;
      schedule_settings.penalty_reduction = gbcl.penalty_reduction;
      schedule_settings.mu_adapt = gbcl.mu_adapt;
      schedule_settings.smallest_penalty = gbcl.mu_min;
      schedule_settings.smallest_equality_penalty = gbcl.mu_min;
      schedule_settings.proximal_slack_reduction = gbcl.rho_p_outer;
      schedule_settings.safe_guard = gbcl.safe_guard;
      schedule_settings.violation_exponent = gbcl.mu_exp;
      schedule_settings.initial_tolerance = gbcl.eps_outer_init;
      schedule_settings.newton_tolerance = gbcl.eps_newton_init;
    } else {
      const auto& gbcl = tuning.gbcl;
      initial_cone_penalty = gbcl.mu_in;
      initial_equality_penalty = gbcl.mu_eq;
      schedule_settings.loosen_exponent = gbcl.alpha;
      schedule_settings.tighten_exponent = gbcl.beta;
      schedule_settings.penalty_reduction = gbcl.penalty_reduction;
      schedule_settings.mu_adapt = gbcl.mu_adapt;
      schedule_settings.smallest_penalty = gbcl.mu_min;
      schedule_settings.smallest_equality_penalty = gbcl.mu_min;
      schedule_settings.initial_tolerance = gbcl.eps_outer_init;
      schedule_settings.newton_tolerance = gbcl.eps_newton_init;
      schedule_settings.violation_exponent = gbcl.mu_exp;
      schedule_settings.proximal_slack_reduction = gbcl.rho_p_outer;
      schedule_settings.safe_guard = gbcl.safe_guard;
    }
  }
  schedule_settings.initial_penalty = initial_cone_penalty;
  schedule_settings.equality_penalty_cap = initial_equality_penalty;
  schedule_settings.smallest_rho = barrier.reg_fine;
  schedule_settings.smallest_proximal_slack = shared.rho_p_min;
  schedule_settings.coarse_floor = barrier.reg_floor;
  schedule_settings.fine_floor = barrier.reg_fine;
  schedule_settings.eps_regularisation = barrier.eps_reg;
  schedule_settings.largest_reduction = 1.0;
  schedule_settings.improvement = barrier.improvement;
  schedule_settings.infeasibility_threshold =
      barrier.infeasibility_threshold;
  schedule_settings.stalled_rate = barrier.stalled_rate;
  schedule_settings.stalls_before_fine = barrier.stall_before_fine;
  schedule_settings.early_outers = barrier.early_outers;
  schedule_settings.schedule_rho = interior;
  auto schedule = make_schedule(
      settings.method == Method::InteriorExp ? ScheduleKind::InteriorExp
      : interior                             ? ScheduleKind::Interior
      : which == Penalty::Bcl                ? ScheduleKind::Bcl
      : which == Penalty::GBclExp            ? ScheduleKind::GbclExp
                                             : ScheduleKind::Gbcl,
      schedule_settings);

  InitialisationSettings initialisation_settings;
  initialisation_settings.seeded = barrier.seeded_start;
  initialisation_settings.pad_constant = barrier.pad_constant;
  initialisation_settings.pad_fraction = barrier.pad_fraction;
  auto initialisation = make_initialisation(smoothing->requires_interior(),
                                            initialisation_settings);

  TerminationSettings termination_settings;
  termination_settings.eps_absolute = settings.eps_abs;
  termination_settings.eps_relative = settings.eps_rel;
  termination_settings.eps_gap_absolute = settings.eps_gap_abs;
  termination_settings.eps_gap_relative = settings.eps_gap_rel;
  termination_settings.eps_primal_infeasible = shared.eps_infeas;
  termination_settings.eps_dual_infeasible = shared.eps_infeas;

  Termination termination;
  termination.setup(cones, columns, rows);

  Iterate iterate;
  iterate.resize(columns, rows);
  iterate.rho = shared.rho;
  iterate.proximal_slack = shared.rho_p;
  iterate.regularisation_floor = 0.0;
  iterate.cone_penalty.setConstant(initial_cone_penalty);
  iterate.equality_penalty.setConstant(initial_equality_penalty);
  for (std::size_t j = 0; j < cones.size(); ++j)
    if (!has_interior(cones[j]))
      iterate.cone_penalty.segment(offsets[j], offsets[j + 1] - offsets[j])
          .setConstant(initial_equality_penalty);

  if (warm && warm->has_point) {
    scale_primal(scaled, warm->x, iterate.x);
    scale_slack(scaled, warm->s, iterate.s);
    scale_dual(scaled, warm->z, iterate.z);
    iterate.y = iterate.z;
    initialisation->warm(cones, iterate);
  } else {
    initialisation->cold(working, cones, *road, iterate);
  }
  schedule->setup(cones, iterate, smoothing->reference(iterate));

  Vector dx(columns), ds(rows), dy(rows), dz(rows);
  Direction direction{dx, ds, dy};
  Vector previous_x = iterate.x, previous_z = iterate.z;
  Vector best_x = iterate.x, best_s = iterate.s, best_z = iterate.z,
         best_y = iterate.y;
  Scalar best_quality = std::numeric_limits<Scalar>::infinity();
  Vector primal_increment(columns), dual_increment(rows);
  Vector unscaled_x(columns), unscaled_s(rows), unscaled_z(rows);

  const Scalar degenerate_step = interior ? barrier.degenerate_step : 0.0;
  const Scalar refactor_ratchet = interior ? barrier.refactor_ratchet : 1.0;
  Scalar written_rho = std::numeric_limits<Scalar>::quiet_NaN();
  Scalar last_primal_step = 0.0, last_dual_step = 0.0;
  for (std::size_t outer = 0; outer < settings.max_iter_outer; ++outer) {
    if (results.inner_iterations >= settings.max_newton) {
      results.status = Status::MaxNewton;
      break;
    }
    ++results.outer_iterations;
    if (!(iterate.rho == written_rho)) {
      road->values(scaled.P, scaled.E, iterate.rho);
      written_rho = iterate.rho;
    }
    bool inner_converged = false;
    std::size_t taken = 0;
    last_primal_step = 0.0;
    last_dual_step = 0.0;
    const std::size_t inner_budget =
        smoothing->one_step_per_outer() ? 1 : settings.max_iter_inner;
    for (std::size_t inner = 0; inner < inner_budget &&
                                 results.inner_iterations < settings.max_newton;
         ++inner) {
      smoothing->residuals(iterate, working);
      if (smoothing->current().largest() <= schedule->inner_tolerance()) {
        inner_converged = true;
        break;
      }
      ++results.inner_iterations;
      ++taken;

      bool factored = false;
      for (std::size_t retry = 0; retry <= shared.max_refactor;
           ++retry) {
        road->assemble(smoothing->blocks(), smoothing->scalings(),
                       iterate.proximal_slack, iterate.equality_penalty);
        if (road->factor()) { factored = true; break; }
        const Scalar bump = refactor_ratchet > 1.0
                                ? shared.refactor_bump * shared.refactor_bump
                                : shared.refactor_bump;
        iterate.proximal_slack =
            std::max(iterate.proximal_slack, iterate.regularisation_floor) * bump;
        iterate.rho *= bump;
        if (refactor_ratchet > 1.0)
          iterate.regularisation_floor =
              std::min(refactor_ratchet * std::max(iterate.regularisation_floor,
                                                   barrier.reg_fine),
                       barrier.eps_reg);
        road->values(scaled.P, scaled.E, iterate.rho);
        written_rho = iterate.rho;
        smoothing->residuals(iterate, working);
      }
      if (!factored) {
        results.status = Status::NumericalFailure;
        unscale_primal(scaled, iterate.x, unscaled_x);
        unscale_dual(scaled, iterate.z, unscaled_z);
        results.kkt = termination.evaluate(
            original, cones, unscaled_x, unscaled_z,
            termination_settings);
        break;
      }

      road->solve(smoothing->road_residuals(), direction);
      smoothing->recover_dual_step(ds, dz);
      for (std::size_t retry = 0;
           retry <= shared.max_refactor &&
           !(dx.allFinite() && ds.allFinite() && dy.allFinite() &&
             dz.allFinite());
           ++retry) {
        const Scalar bump = refactor_ratchet > 1.0
                                ? shared.refactor_bump * shared.refactor_bump
                                : shared.refactor_bump;
        iterate.proximal_slack =
            std::max(iterate.proximal_slack, iterate.regularisation_floor) *
            bump;
        iterate.rho *= bump;
        if (refactor_ratchet > 1.0)
          iterate.regularisation_floor =
              std::min(refactor_ratchet * std::max(iterate.regularisation_floor,
                                                   barrier.reg_fine),
                       barrier.eps_reg);
        road->values(scaled.P, scaled.E, iterate.rho);
        written_rho = iterate.rho;
        smoothing->residuals(iterate, working);
        road->assemble(smoothing->blocks(), smoothing->scalings(),
                       iterate.proximal_slack, iterate.equality_penalty);
        if (!road->factor()) continue;
        road->solve(smoothing->road_residuals(), direction);
        smoothing->recover_dual_step(ds, dz);
      }
      if (!dx.allFinite() || !ds.allFinite() || !dy.allFinite() ||
          !dz.allFinite())
        break;
      StepPair step =
          globalisation->step(working, cones, iterate, dx, ds, dz);
      for (std::size_t retry = 0;
           retry <= shared.max_refactor && degenerate_step > 0.0 &&
           std::min(step.primal, step.dual) < degenerate_step;
           ++retry) {
        const Scalar bump = refactor_ratchet > 1.0
                                ? shared.refactor_bump * shared.refactor_bump
                                : shared.refactor_bump;
        iterate.proximal_slack =
            std::max(iterate.proximal_slack, iterate.regularisation_floor) *
            bump;
        iterate.rho *= bump;
        if (refactor_ratchet > 1.0)
          iterate.regularisation_floor =
              std::min(refactor_ratchet * std::max(iterate.regularisation_floor,
                                                   barrier.reg_fine),
                       barrier.eps_reg);
        road->values(scaled.P, scaled.E, iterate.rho);
        written_rho = iterate.rho;
        smoothing->residuals(iterate, working);
        road->assemble(smoothing->blocks(), smoothing->scalings(),
                       iterate.proximal_slack, iterate.equality_penalty);
        if (!road->factor()) continue;
        road->solve(smoothing->road_residuals(), direction);
        smoothing->recover_dual_step(ds, dz);
        if (!dx.allFinite() || !ds.allFinite()) continue;
        step = globalisation->step(working, cones, iterate, dx, ds, dz);
      }

      if (smoothing->corrector(iterate, ds, dz, step.primal, step.dual)) {
        road->solve(smoothing->road_residuals(), direction);
        smoothing->recover_dual_step(ds, dz);
        step = globalisation->step(working, cones, iterate, dx, ds, dz);
      }

      if (!std::isfinite(step.primal) || !std::isfinite(step.dual)) break;
      last_primal_step = step.primal;
      last_dual_step = step.dual;
      iterate.x += step.primal * dx;
      iterate.s += step.primal * ds;
      iterate.y += step.dual * dy;
      iterate.z += step.dual * dz;
      if (!iterate.x.allFinite() || !iterate.s.allFinite() ||
          !iterate.y.allFinite() || !iterate.z.allFinite()) {
        iterate.x = best_x;
        iterate.s = best_s;
        iterate.z = best_z;
        iterate.y = best_y;
        break;
      }
      if (shared.rho_p_decay < 1.0)
        iterate.proximal_slack = std::max(
            iterate.proximal_slack * shared.rho_p_decay, shared.rho_p_min);
      if (step.primal * std::max(infinity_norm(dx), infinity_norm(ds)) <
          shared.stall_tol)
        break;
    }
    if (results.status == Status::NumericalFailure) break;

    unscale_primal(scaled, iterate.x, unscaled_x);
    unscale_dual(scaled, iterate.z, unscaled_z);
    results.kkt = termination.evaluate(original, cones, unscaled_x, unscaled_z,
                                       termination_settings);
    if (settings.verbose)
      std::printf(
          "  outer %3zu pri %.2e dua %.2e comp %.2e | inner %2zu res %.2e "
          "tol %.2e rho %.2e rp %.2e mu %.2e ap %.3f ad %.3f sig %.2e\n",
          outer, results.kkt.primal_residual, results.kkt.dual_residual,
          results.kkt.complementarity, taken,
          smoothing->current().largest(),
          schedule->inner_tolerance(), iterate.rho,
          iterate.proximal_slack,
          smallest_entry(iterate.cone_penalty, 0.0), last_primal_step,
          last_dual_step, smoothing->centring());
    const Scalar quality =
        std::max({results.kkt.primal_residual, results.kkt.dual_residual,
                  results.kkt.dual_cone_violation,
                  results.kkt.complementarity});
    if (std::isfinite(quality) && quality < best_quality) {
      best_quality = quality;
      best_x = iterate.x;
      best_s = iterate.s;
      best_z = iterate.z;
      best_y = iterate.y;
    }
    if (results.kkt.converged) {
      results.status = Status::Solved;
      break;
    }

    primal_increment = iterate.x - previous_x;
    dual_increment = iterate.z - previous_z;
    previous_x = iterate.x;
    previous_z = iterate.z;
    if (termination.primal_infeasible(working, cones, dual_increment,
                                      termination_settings)) {
      results.status = Status::PrimalInfeasible;
      break;
    }
    if (termination.dual_infeasible(working, cones, primal_increment,
                                    termination_settings)) {
      results.status = Status::DualInfeasible;
      break;
    }

    schedule->update(iterate, termination.row_violation(),
                     smoothing->reference(iterate),
                     results.kkt.primal_residual, results.kkt.dual_residual,
                     inner_converged, taken);
  }

  if (results.status != Status::Solved) {
    const Scalar reached =
        std::max({results.kkt.primal_residual, results.kkt.dual_residual,
                  results.kkt.dual_cone_violation,
                  results.kkt.complementarity});
    if (!std::isfinite(reached) || reached > best_quality) {
      iterate.x = best_x;
      iterate.s = best_s;
      iterate.z = best_z;
      iterate.y = best_y;
      unscale_primal(scaled, best_x, unscaled_x);
      unscale_dual(scaled, best_z, unscaled_z);
      results.kkt = termination.evaluate(original, cones, unscaled_x,
                                         unscaled_z,
                                         termination_settings);
      if (results.kkt.converged) results.status = Status::Solved;
    }
  }
  unscale_primal(scaled, iterate.x, results.x);
  unscale_dual(scaled, iterate.z, results.z);
  results.s.noalias() = E * results.x;
  results.s = b - results.s;
  results.has_point = true;
  results.seconds =
      std::chrono::duration<Scalar>(Clock::now() - started).count();
  return results;
}

}
