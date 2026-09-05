#include "schedule/schedule.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace proxgqp {

namespace {

void flatten_on_curved(const Cones& cones, const std::vector<Index>& offsets,
                       Vector& values) {
  for (std::size_t j = 0; j < cones.size(); ++j) {
    if (!is_curved(cones[j])) continue;
    const Index start = offsets[j];
    const Index length = offsets[j + 1] - offsets[j];
    values.segment(start, length)
        .setConstant(values.segment(start, length).minCoeff());
  }
}


struct Bcl final : Schedule {
  explicit Bcl(const ScheduleSettings& given) : settings(given) {}

  void setup(const Cones&, const Iterate& iterate, Scalar) override {
    penalty = iterate.cone_penalty.size() ? iterate.cone_penalty.maxCoeff()
                                          : settings.initial_penalty;
    inner_floor = std::min(settings.eps_absolute, Scalar(1e-9));
    outer_bar_init = std::pow(Scalar(0.1), settings.loosen_exponent);
    outer_bar = outer_bar_init;
    inner_bar = 1.0;
    accepted.setConstant(iterate.s.size(), outer_bar);
    y_previous = iterate.y;
    z_previous = iterate.z;
    previous_primal = std::numeric_limits<Scalar>::infinity();
    previous_dual = std::numeric_limits<Scalar>::infinity();
    taken = 0;
  }

  void update(Iterate& iterate, const Vector&, Scalar, Scalar primal_residual,
              Scalar dual_residual, bool, std::size_t) override {
    penalty = iterate.cone_penalty.size() ? iterate.cone_penalty.maxCoeff()
                                          : settings.initial_penalty;
    if (primal_residual <= outer_bar || taken > settings.safe_guard) {
      iterate.x_centre = iterate.x;
      iterate.s_centre = iterate.s;
      iterate.z_centre = iterate.z;
      iterate.y_centre = iterate.y;
      outer_bar *= std::pow(penalty, settings.tighten_exponent);
      inner_bar = std::max(inner_bar * penalty, inner_floor);
    } else {
      if (settings.revert_on_reject) {
        iterate.y = y_previous;
        iterate.z = z_previous;
      }
      iterate.cone_penalty =
          (iterate.cone_penalty * settings.penalty_reduction)
              .cwiseMax(settings.smallest_penalty);
      iterate.equality_penalty =
          (iterate.equality_penalty * settings.penalty_reduction)
              .cwiseMax(settings.smallest_equality_penalty);
      penalty = iterate.cone_penalty.maxCoeff();
      outer_bar = outer_bar_init * std::pow(penalty, settings.loosen_exponent);
      inner_bar = std::max(penalty, inner_floor);
    }

    if (primal_residual >= previous_primal && dual_residual >= previous_dual &&
        penalty <= settings.restart_threshold) {
      iterate.cone_penalty.setConstant(settings.restart_penalty);
      iterate.equality_penalty.setConstant(settings.restart_penalty);
    }
    previous_primal = primal_residual;
    previous_dual = dual_residual;
    y_previous = iterate.y;
    z_previous = iterate.z;
    accepted.setConstant(accepted.size(), outer_bar);
    ++taken;

    if (settings.proximal_slack_reduction < 1.0)
      iterate.proximal_slack =
          std::max(settings.smallest_proximal_slack,
                   iterate.proximal_slack * settings.proximal_slack_reduction);
  }

  const Vector& tolerance() const override { return accepted; }
  Scalar inner_tolerance() const override { return inner_bar; }
  const char* name() const override { return "bcl"; }

  ScheduleSettings settings;
  Vector accepted, y_previous, z_previous;
  Scalar penalty = 0.0, inner_floor = 0.0;
  Scalar outer_bar = 0.0, outer_bar_init = 0.0, inner_bar = 0.0;
  Scalar previous_primal = std::numeric_limits<Scalar>::infinity();
  Scalar previous_dual = std::numeric_limits<Scalar>::infinity();
  std::size_t taken = 0;
};

struct Gbcl final : Schedule {
  explicit Gbcl(const ScheduleSettings& given) : settings(given) {}

  void setup(const Cones& given_cones, const Iterate& iterate,
             Scalar) override {
    cones = &given_cones;
    offsets = block_offsets(given_cones);
    factor.setOnes(iterate.s.size());
    largest_penalty = iterate.cone_penalty.size()
                          ? iterate.cone_penalty.maxCoeff()
                          : settings.initial_penalty;
    penalty_floor = settings.smallest_penalty;
    cut = settings.penalty_reduction;
    bar = settings.initial_tolerance *
          std::pow(largest_penalty, settings.loosen_exponent);
    inner = settings.newton_tolerance * largest_penalty;
    accepted.setConstant(iterate.s.size(), bar);
    taken = 0;
    previous_efficiency = -std::numeric_limits<Scalar>::infinity();
    previous_violation = 0.0;
    direction = -1.0;
    have_previous = false;
  }

  void adapt(Scalar primal_residual, std::size_t inner_taken) {
    if (have_previous && inner_taken > 0 && primal_residual > 0.0) {
      const Scalar tiny = 1e-300;
      const Scalar efficiency =
          std::log10(std::max(previous_violation, tiny) /
                     std::max(primal_residual, tiny)) /
          static_cast<Scalar>(inner_taken);
      if (efficiency < previous_efficiency) direction = -direction;
      previous_efficiency = efficiency;
      cut = std::min(
          std::max(cut * std::pow(settings.mu_adapt, direction), Scalar(1e-3)),
          Scalar(0.9));
    }
    previous_violation = primal_residual;
    have_previous = true;
  }

  void update(Iterate& iterate, const Vector& violation, Scalar,
              Scalar primal_residual, Scalar, bool inner_converged,
              std::size_t inner_taken) override {
    const Index rows = violation.size();
    const Scalar theta = settings.penalty_reduction;

    for (Index row = 0; row < rows; ++row) {
      if (violation(row) > accepted(row)) continue;
      iterate.s_centre(row) = iterate.s(row);
      iterate.z_centre(row) = iterate.z(row);
      iterate.y_centre(row) = iterate.y(row);
    }

    if (!inner_converged) {
      penalty_floor = std::max(penalty_floor,
                               std::min(largest_penalty / theta,
                                        settings.initial_penalty));
      for (Index row = 0; row < rows; ++row) {
        if (violation(row) <= accepted(row)) continue;
        iterate.cone_penalty(row) = std::min(iterate.cone_penalty(row) / theta,
                                             settings.initial_penalty);
        iterate.equality_penalty(row) =
            std::min(iterate.equality_penalty(row) / theta,
                     settings.equality_penalty_cap);
      }
    } else {
      adapt(primal_residual, inner_taken);
      const Scalar low = std::max(penalty_floor, settings.smallest_penalty);
      for (Index row = 0; row < rows; ++row) {
        const Scalar ratio = violation(row) / std::max(accepted(row), 1e-300);
        factor(row) = std::min(
            Scalar(1), std::max(cut, std::pow(std::max(ratio, 1e-300),
                                              -settings.violation_exponent)));
      }
      flatten_on_curved(*cones, offsets, factor);
      iterate.cone_penalty =
          iterate.cone_penalty.cwiseProduct(factor).cwiseMax(low);
      iterate.equality_penalty =
          iterate.equality_penalty.cwiseProduct(factor).cwiseMax(
              settings.smallest_penalty);
    }

    for (Index row = 0; row < rows; ++row)
      accepted(row) =
          violation(row) <= accepted(row)
              ? std::max(accepted(row) * std::pow(iterate.cone_penalty(row),
                                                  settings.tighten_exponent),
                         settings.eps_absolute)
              : settings.initial_tolerance *
                    std::pow(iterate.cone_penalty(row),
                             settings.loosen_exponent);

    largest_penalty = iterate.cone_penalty.size()
                          ? iterate.cone_penalty.maxCoeff()
                          : settings.initial_penalty;
    if (primal_residual <= bar || taken >= settings.safe_guard) {
      iterate.x_centre = iterate.x;
      inner = std::max(inner * largest_penalty, settings.smallest_tolerance);
      bar = std::max(bar * std::pow(largest_penalty, settings.tighten_exponent),
                     settings.eps_absolute);
    } else {
      inner = inner_taken == 0
                  ? std::max(inner * largest_penalty, settings.smallest_tolerance)
                  : settings.newton_tolerance * largest_penalty;
      bar = settings.initial_tolerance *
            std::pow(largest_penalty, settings.loosen_exponent);
    }
    ++taken;

    if (settings.proximal_slack_reduction < 1.0)
      iterate.proximal_slack =
          std::max(settings.smallest_proximal_slack,
                   iterate.proximal_slack * settings.proximal_slack_reduction);
  }

  const Vector& tolerance() const override { return accepted; }
  Scalar inner_tolerance() const override { return inner; }
  const char* name() const override { return "gbcl"; }

  ScheduleSettings settings;
  const Cones* cones = nullptr;
  std::vector<Index> offsets;
  Vector accepted, factor;
  Scalar largest_penalty = 0.0, penalty_floor = 0.0, cut = 0.0;
  Scalar bar = 0.0, inner = 0.0;
  Scalar previous_efficiency = 0.0, previous_violation = 0.0, direction = -1.0;
  std::size_t taken = 0;
  bool have_previous = false;
};


struct Interior final : Schedule {
  explicit Interior(const ScheduleSettings& given) : settings(given) {}

  void setup(const Cones& given_cones, const Iterate& iterate,
             Scalar reference) override {
    cones = &given_cones;
    offsets = block_offsets(given_cones);
    accepted.setConstant(iterate.s.size(), settings.smallest_tolerance);
    limit = settings.coarse_floor;
    previous_measure = reference;
    previous_primal = previous_dual = std::numeric_limits<Scalar>::infinity();
    without_primal = without_dual = 0;
    taken = 0;
  }

  void update(Iterate& iterate, const Vector&, Scalar reference,
              Scalar primal_residual, Scalar dual_residual, bool,
              std::size_t) override {
    const Scalar progress =
        reference > 0.0
            ? reference
            : std::max(std::max(primal_residual, dual_residual), Scalar(0));
    const Scalar rate =
        previous_measure > 0.0 && progress > 0.0
            ? std::max(Scalar(0), (previous_measure - progress) / previous_measure)
            : 0.0;
    const Scalar accepted_factor =
        std::min(1.0 - rate, settings.largest_reduction);
    const Scalar stalled_factor =
        std::min(1.0 - settings.stalled_rate * rate,
                 std::sqrt(settings.largest_reduction));
    previous_measure = progress;

    const Scalar dual_prox =
        iterate.rho * infinity_norm(Vector(iterate.x - iterate.x_centre));
    const Scalar primal_prox =
        iterate.proximal_slack *
        infinity_norm(Vector(iterate.s - iterate.s_centre));
    const Scalar bar = settings.infeasibility_threshold;
    const bool at_fine = limit == settings.fine_floor;
    const Scalar guarded = std::max(limit, iterate.regularisation_floor);

    if (dual_residual < settings.improvement * previous_dual ||
        dual_residual < settings.eps_regularisation ||
        (at_fine && dual_prox < bar)) {
      iterate.x_centre = iterate.x;
      iterate.rho = std::max(guarded, accepted_factor * iterate.rho);
    } else {
      ++without_primal;
      if (taken < settings.early_outers || dual_prox < bar)
        iterate.rho = std::max(guarded, stalled_factor * iterate.rho);
    }

    if (primal_residual < settings.improvement * previous_primal ||
        primal_residual < settings.eps_regularisation ||
        (at_fine && primal_prox < bar)) {
      iterate.s_centre = iterate.s;
      iterate.y_centre = iterate.y;
      iterate.z_centre = iterate.z;
      relax(iterate, accepted_factor);
    } else {
      ++without_dual;
      if (taken < settings.early_outers || primal_prox < bar)
        relax(iterate, stalled_factor);
    }

    if (!at_fine &&
        ((without_primal > settings.stalls_before_fine &&
          iterate.rho == limit) ||
         (without_dual > settings.stalls_before_fine &&
          iterate.proximal_slack == limit)) &&
        dual_prox < bar && primal_prox < bar) {
      limit = settings.fine_floor;
      without_primal = without_dual = 0;
    }

    previous_primal = primal_residual;
    previous_dual = dual_residual;
    ++taken;
  }

  const Vector& tolerance() const override { return accepted; }
  Scalar inner_tolerance() const override { return settings.smallest_tolerance; }
  const char* name() const override { return "interior"; }

 private:
  void relax(Iterate& iterate, Scalar factor) {
    const Scalar guarded = std::max(limit, iterate.regularisation_floor);
    iterate.proximal_slack =
        std::max(guarded, iterate.proximal_slack * factor);
    iterate.equality_penalty =
        (iterate.equality_penalty * factor).cwiseMax(guarded);
    for (std::size_t j = 0; j < cones->size(); ++j) {
      if (has_interior((*cones)[j])) continue;
      const Index start = offsets[j];
      const Index length = offsets[j + 1] - offsets[j];
      iterate.cone_penalty.segment(start, length) =
          (iterate.cone_penalty.segment(start, length) * factor)
              .cwiseMax(guarded);
    }
  }

  ScheduleSettings settings;
  const Cones* cones = nullptr;
  std::vector<Index> offsets;
  Vector accepted;
  Scalar limit = 1e-10;
  Scalar previous_measure = 0.0, previous_primal = 0.0, previous_dual = 0.0;
  std::size_t without_primal = 0, without_dual = 0, taken = 0;
};

}  // namespace

std::unique_ptr<Schedule> make_schedule(ScheduleKind kind,
                                        const ScheduleSettings& settings) {
  switch (kind) {
    case ScheduleKind::Bcl: return std::make_unique<Bcl>(settings);
    case ScheduleKind::Interior: return std::make_unique<Interior>(settings);
    default: return std::make_unique<Gbcl>(settings);
  }
}

}  // namespace proxgqp
