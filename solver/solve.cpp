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

void cone_violation(const Cones& cones, const std::vector<Index>& offsets,
                    const Vector& slack, Vector& work, Vector& out) {
  work = -slack;
  evaluate_projection_all(cones, offsets, work, out);
  out = out.cwiseAbs();
  for (std::size_t j = 0; j < cones.size(); ++j) {
    if (!is_curved(cones[j])) continue;
    const Index start = offsets[j];
    const Index length = offsets[j + 1] - offsets[j];
    out.segment(start, length)
        .setConstant(out.segment(start, length).lpNorm<Eigen::Infinity>());
  }
}

}

Results solve(const SparseMatrix& P, const Vector& q, const SparseMatrix& E,
              const Vector& f, const Cones& cones, const Settings& settings,
              const Results* warm) {
  const auto started = Clock::now();
  Scalar phase_residuals = 0.0, phase_assemble = 0.0, phase_factor = 0.0,
         phase_solve = 0.0, phase_step = 0.0;
  auto tick = [] { return Clock::now(); };
  auto tock = [](const std::chrono::time_point<Clock>& from) {
    return std::chrono::duration<Scalar>(Clock::now() - from).count();
  };
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
    equilibrate(P, q, E, f, cones, ruiz_settings, scaled);
  } else {
    scaled.P = P; scaled.q = q; scaled.E = E; scaled.f = f;
    scaled.column_scale.setOnes(columns);
    scaled.row_scale.setOnes(rows);
    scaled.cost_scale = 1.0;
  }
  const ProblemData working{&scaled.P, &scaled.q, &scaled.E, &scaled.f};
  const ProblemData original{&P, &q, &E, &f};

  const Tuning& tuning = settings.tuning;
  const bool interior = settings.method == Method::Interior;
  const Shared& shared = tuning.shared(settings.method);

  RoadSettings road_settings;
  road_settings.backend = settings.backend;
  road_settings.refinement_budget = shared.refine;
  road_settings.max_threads = settings.max_threads;
  auto road = make_road(settings.road, road_settings);
  road->setup(scaled.P, scaled.E, cones);

  SmoothingSettings smoothing_settings;
  smoothing_settings.mehrotra = tuning.interior.mehrotra;
  smoothing_settings.centring_fraction = tuning.interior.sigma_centre;
  smoothing_settings.smallest_centring = tuning.interior.sigma_min;
  smoothing_settings.largest_centring = tuning.interior.sigma_max;
  auto smoothing = make_smoothing(
      interior ? SmoothingKind::Barrier : SmoothingKind::Projection,
      smoothing_settings);
  smoothing->setup(cones, columns, rows);

  GlobalisationSettings globalisation_settings;
  globalisation_settings.boundary_fraction = tuning.interior.tau;
  globalisation_settings.curved_boundary_fraction = tuning.interior.tau_curved;
  globalisation_settings.largest_secant = tuning.semismooth.ls_max_secant;
  auto globalisation = make_globalisation(
      interior ? GlobalisationKind::FractionToBoundary
               : GlobalisationKind::ExactSearch,
      globalisation_settings);
  globalisation->setup(cones, columns, rows);

  ScheduleSettings schedule_settings;
  schedule_settings.schedule_rho = interior;
  schedule_settings.penalty_reduction = tuning.semismooth.penalty_reduction;
  schedule_settings.smallest_penalty = tuning.semismooth.mu_min;
  schedule_settings.initial_tolerance = tuning.semismooth.eps_outer_init;
  schedule_settings.tighten_exponent = tuning.semismooth.beta_bcl;
  schedule_settings.loosen_exponent = tuning.semismooth.alpha_bcl;
  schedule_settings.violation_exponent = tuning.semismooth.mu_exp;
  schedule_settings.smallest_factor = tuning.semismooth.mu_cut;
  schedule_settings.smallest_tolerance = tuning.semismooth.smallest_tolerance;
  schedule_settings.smallest_rho = tuning.interior.reg_fine;
  schedule_settings.smallest_proximal_slack = shared.rho_p_min;
  schedule_settings.coarse_floor = tuning.interior.reg_floor;
  schedule_settings.fine_floor = tuning.interior.reg_fine;
  schedule_settings.eps_regularisation = tuning.interior.eps_reg;
  schedule_settings.improvement = tuning.interior.improvement;
  schedule_settings.infeasibility_threshold =
      tuning.interior.infeasibility_threshold;
  schedule_settings.stalled_rate = tuning.interior.stalled_rate;
  schedule_settings.stalls_before_fine = tuning.interior.stall_before_fine;
  schedule_settings.early_outers = tuning.interior.early_outers;
  auto schedule = make_schedule(
      interior ? ScheduleKind::Interior
               : (tuning.semismooth.penalty == Penalty::Bcl
                      ? ScheduleKind::Bcl
                      : ScheduleKind::Gbcl),
      schedule_settings);

  InitialisationSettings initialisation_settings;
  initialisation_settings.seeded = tuning.interior.seeded_start;
  initialisation_settings.pad_constant = tuning.interior.pad_constant;
  initialisation_settings.pad_fraction = tuning.interior.pad_fraction;
  auto initialisation = make_initialisation(smoothing->requires_interior(),
                                            initialisation_settings);

  TerminationSettings termination_settings;
  termination_settings.eps_absolute = settings.eps_abs;
  termination_settings.eps_relative = settings.eps_rel;
  termination_settings.eps_primal_infeasible = shared.eps_infeas;
  termination_settings.eps_dual_infeasible = shared.eps_infeas;

  Termination termination;
  termination.setup(cones, columns, rows);

  Iterate iterate;
  iterate.resize(columns, rows);
  iterate.rho = shared.rho;
  iterate.proximal_slack = shared.rho_p;
  iterate.cone_penalty.setConstant(tuning.semismooth.mu_in);
  iterate.equality_penalty.setConstant(shared.mu_eq);
  for (std::size_t j = 0; j < cones.size(); ++j)
    if (!has_interior(cones[j]))
      iterate.cone_penalty.segment(offsets[j], offsets[j + 1] - offsets[j])
          .setConstant(shared.mu_eq);

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
  Vector violation(rows), work_rows(rows);
  Vector previous_x = iterate.x, previous_z = iterate.z;
  Vector best_x = iterate.x, best_s = iterate.s, best_z = iterate.z,
         best_y = iterate.y;
  Scalar best_quality = std::numeric_limits<Scalar>::infinity();
  Vector primal_increment(columns), dual_increment(rows);
  Vector unscaled_x(columns), unscaled_s(rows), unscaled_z(rows);

  Scalar written_rho = std::numeric_limits<Scalar>::quiet_NaN();
  for (std::size_t outer = 0; outer < settings.max_iter; ++outer) {
    ++results.outer_iterations;
    if (!(iterate.rho == written_rho)) {
      road->values(scaled.P, scaled.E, iterate.rho);
      written_rho = iterate.rho;
    }
    bool inner_converged = false;
    std::size_t taken = 0;
    const std::size_t inner_budget =
        smoothing->one_step_per_outer() ? 1 : settings.max_inner;
    for (std::size_t inner = 0; inner < inner_budget; ++inner) {
      ++results.inner_iterations;
      ++taken;
      { const auto mark = tick(); smoothing->residuals(iterate, working); phase_residuals += tock(mark); }
      if (smoothing->current().largest() <=
          smallest_entry(schedule->tolerance(), 0.0)) {
        inner_converged = true;
        break;
      }

      bool factored = false;
      for (std::size_t retry = 0; retry <= shared.max_refactor;
           ++retry) {
        { const auto mark = tick(); road->assemble(smoothing->blocks(), smoothing->scalings(),
                       iterate.proximal_slack, iterate.equality_penalty); phase_assemble += tock(mark); }
        { const auto mark = tick(); const bool ok = road->factor(); phase_factor += tock(mark);
          if (ok) { factored = true; break; } }
        iterate.proximal_slack *= shared.refactor_bump;
        iterate.rho *= shared.refactor_bump;
        road->values(scaled.P, scaled.E, iterate.rho);
        written_rho = iterate.rho;
        smoothing->residuals(iterate, working);
      }
      if (!factored) {
        results.status = Status::NumericalFailure;
        outer = settings.max_iter;
        unscale_primal(scaled, iterate.x, unscaled_x);
        unscale_dual(scaled, iterate.z, unscaled_z);
        results.kkt = termination.evaluate(
            original, cones, unscaled_x, unscaled_z,
            termination_settings);
        break;
      }

      { const auto mark = tick(); road->solve(smoothing->road_residuals(), direction); phase_solve += tock(mark); }
      smoothing->recover_dual_step(ds, dz);
      if (!dx.allFinite() || !ds.allFinite() || !dy.allFinite() ||
          !dz.allFinite())
        break;
      const auto step_mark = tick();
      StepPair step =
          globalisation->step(working, cones, iterate, dx, ds, dz);
      phase_step += tock(step_mark);

      if (smoothing->corrector(iterate, ds, dz, step.primal, step.dual)) {
        road->solve(smoothing->road_residuals(), direction);
        smoothing->recover_dual_step(ds, dz);
        step = globalisation->step(working, cones, iterate, dx, ds, dz);
      }

      if (!std::isfinite(step.primal) || !std::isfinite(step.dual)) break;
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
    }
    if (results.status == Status::NumericalFailure) break;

    unscale_primal(scaled, iterate.x, unscaled_x);
    unscale_dual(scaled, iterate.z, unscaled_z);
    results.kkt = termination.evaluate(original, cones, unscaled_x, unscaled_z,
                                       termination_settings);
    if (settings.verbose)
      std::printf(
          "  outer %3zu pri %.2e dua %.2e comp %.2e | inner %2zu res %.2e "
          "tol %.2e rho %.2e rp %.2e mu %.2e\n",
          outer, results.kkt.primal_residual, results.kkt.dual_residual,
          results.kkt.complementarity, taken,
          smoothing->current().largest(),
          smallest_entry(schedule->tolerance(), 0.0), iterate.rho,
          iterate.proximal_slack,
          smallest_entry(iterate.cone_penalty, 0.0));
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

    cone_violation(cones, offsets, iterate.s, work_rows, violation);
    schedule->update(iterate, violation, smoothing->reference(iterate),
                     results.kkt.primal_residual, results.kkt.dual_residual,
                     inner_converged);
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
  results.s = f - results.s;
  results.has_point = true;
  results.seconds =
      std::chrono::duration<Scalar>(Clock::now() - started).count();
  if (settings.verbose)
    std::printf(
        "  phases: residuals %.4fs  assemble %.4fs  factor %.4fs  solve %.4fs"
        "  step %.4fs  total %.4fs\n",
        phase_residuals, phase_assemble, phase_factor, phase_solve, phase_step,
        results.seconds);
  return results;
}

}
