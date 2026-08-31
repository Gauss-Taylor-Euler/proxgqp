#include "schedule/schedule.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace proxgqp {

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

namespace {

struct Bcl final : Schedule {
  explicit Bcl(const ScheduleSettings& given) : settings(given) {}

  void setup(const Cones& given_cones, const Iterate& iterate,
             Scalar reference) override {
    cones = &given_cones;
    offsets = block_offsets(given_cones);
    accepted.setConstant(iterate.s.size(), settings.initial_tolerance);
    previous_reference = reference;
  }

  void update(Iterate& iterate, const Vector& violation, Scalar reference,
              Scalar, Scalar, bool inner_converged) override {
    const Scalar worst = infinity_norm(violation);
    const Scalar bar = accepted.size() ? accepted(0) : settings.initial_tolerance;
    const Scalar penalty =
        iterate.cone_penalty.size() ? iterate.cone_penalty(0) : Scalar(1);
    if (inner_converged) {
      iterate.x_centre = iterate.x;
      iterate.s_centre = iterate.s;
      iterate.z_centre = iterate.z;
      iterate.y_centre = iterate.y;
      if (settings.schedule_rho)
        iterate.rho = std::max(settings.smallest_rho,
                               iterate.rho * settings.rho_reduction);
      iterate.proximal_slack =
          std::max(settings.smallest_proximal_slack,
                   iterate.proximal_slack * settings.proximal_slack_reduction);
    }
    if (worst <= bar) {
      accepted.setConstant(
          std::max(bar * std::pow(penalty, settings.tighten_exponent),
                   settings.smallest_tolerance));
    } else {
      iterate.cone_penalty =
          (iterate.cone_penalty * settings.penalty_reduction)
              .cwiseMax(settings.smallest_penalty);
      iterate.equality_penalty =
          (iterate.equality_penalty * settings.penalty_reduction)
              .cwiseMax(settings.smallest_penalty);
      accepted.setConstant(
          std::max(settings.smallest_tolerance,
                   settings.initial_tolerance *
                       std::pow(penalty, settings.loosen_exponent)));
    }
    const Scalar rate =
        previous_reference > 0.0
            ? std::max(Scalar(0), (previous_reference - reference) /
                                      previous_reference)
            : 0.0;
    previous_reference = reference;
    if (settings.schedule_rho)
      iterate.rho = std::max(settings.smallest_rho, (1.0 - rate) * iterate.rho);
    iterate.proximal_slack = std::max(settings.smallest_proximal_slack,
                                      (1.0 - rate) * iterate.proximal_slack);
  }

  const Vector& tolerance() const override { return accepted; }
  const char* name() const override { return "bcl"; }

  ScheduleSettings settings;
  const Cones* cones = nullptr;
  std::vector<Index> offsets;
  Vector accepted;
  Scalar previous_reference = 0.0;
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
              Scalar primal_residual, Scalar dual_residual, bool) override {
    const Scalar rate =
        previous_measure > 0.0
            ? std::max(Scalar(0),
                       (previous_measure - reference) / previous_measure)
            : 0.0;
    previous_measure = reference;

    const Scalar dual_prox =
        iterate.rho * infinity_norm(Vector(iterate.x - iterate.x_centre));
    const Scalar primal_prox =
        iterate.proximal_slack *
        infinity_norm(Vector(iterate.s - iterate.s_centre));
    const Scalar bar = settings.infeasibility_threshold;
    const bool at_fine = limit == settings.fine_floor;

    if (dual_residual < settings.improvement * previous_dual ||
        dual_residual < settings.eps_regularisation ||
        (at_fine && dual_prox < bar)) {
      iterate.x_centre = iterate.x;
      iterate.rho = std::max(limit, (1.0 - rate) * iterate.rho);
    } else {
      ++without_primal;
      if (taken < settings.early_outers || dual_prox < bar)
        iterate.rho =
            std::max(limit, (1.0 - settings.stalled_rate * rate) * iterate.rho);
    }

    if (primal_residual < settings.improvement * previous_primal ||
        primal_residual < settings.eps_regularisation ||
        (at_fine && primal_prox < bar)) {
      iterate.s_centre = iterate.s;
      iterate.y_centre = iterate.y;
      iterate.z_centre = iterate.z;
      relax(iterate, 1.0 - rate);
    } else {
      ++without_dual;
      if (taken < settings.early_outers || primal_prox < bar)
        relax(iterate, 1.0 - settings.stalled_rate * rate);
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
  const char* name() const override { return "interior"; }

 private:
  void relax(Iterate& iterate, Scalar factor) {
    iterate.proximal_slack = std::max(limit, iterate.proximal_slack * factor);
    iterate.equality_penalty =
        (iterate.equality_penalty * factor).cwiseMax(limit);
    for (std::size_t j = 0; j < cones->size(); ++j) {
      if (has_interior((*cones)[j])) continue;
      const Index start = offsets[j];
      const Index length = offsets[j + 1] - offsets[j];
      iterate.cone_penalty.segment(start, length) =
          (iterate.cone_penalty.segment(start, length) * factor).cwiseMax(limit);
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

struct Gbcl final : Schedule {
  explicit Gbcl(const ScheduleSettings& given) : settings(given) {}

  void setup(const Cones& given_cones, const Iterate& iterate,
             Scalar reference) override {
    cones = &given_cones;
    offsets = block_offsets(given_cones);
    accepted.setConstant(iterate.s.size(), settings.initial_tolerance);
    factor.setOnes(iterate.s.size());
    previous_reference = reference;
  }

  void update(Iterate& iterate, const Vector& violation, Scalar reference,
              Scalar, Scalar, bool inner_converged) override {
    const Index rows = violation.size();
    for (Index row = 0; row < rows; ++row) {
      const Scalar ratio = violation(row) / std::max(accepted(row), 1e-300);
      factor(row) = std::min(
          Scalar(1), std::max(settings.smallest_factor,
                              std::pow(std::max(ratio, 1e-300),
                                       -settings.violation_exponent)));
    }
    flatten_on_curved(*cones, offsets, factor);

    for (Index row = 0; row < rows; ++row) {
      if (violation(row) > accepted(row)) continue;
      iterate.s_centre(row) = iterate.s(row);
      iterate.z_centre(row) = iterate.z(row);
      iterate.y_centre(row) = iterate.y(row);
    }
    if (inner_converged) {
      iterate.x_centre = iterate.x;
      iterate.s_centre = iterate.s;
      iterate.z_centre = iterate.z;
      iterate.y_centre = iterate.y;
      if (settings.schedule_rho)
        iterate.rho = std::max(settings.smallest_rho,
                               iterate.rho * settings.rho_reduction);
      iterate.proximal_slack =
          std::max(settings.smallest_proximal_slack,
                   iterate.proximal_slack * settings.proximal_slack_reduction);
    }

    iterate.cone_penalty =
        iterate.cone_penalty.cwiseProduct(factor).cwiseMax(
            settings.smallest_penalty);
    iterate.equality_penalty =
        iterate.equality_penalty.cwiseProduct(factor).cwiseMax(
            settings.smallest_penalty);
    for (Index row = 0; row < rows; ++row)
      accepted(row) = std::max(
          settings.smallest_tolerance,
          accepted(row) * std::pow(iterate.cone_penalty(row),
                                   settings.tighten_exponent));

    const Scalar rate =
        previous_reference > 0.0
            ? std::max(Scalar(0), (previous_reference - reference) /
                                      previous_reference)
            : 0.0;
    previous_reference = reference;
    if (settings.schedule_rho)
      iterate.rho = std::max(settings.smallest_rho, (1.0 - rate) * iterate.rho);
    iterate.proximal_slack = std::max(settings.smallest_proximal_slack,
                                      (1.0 - rate) * iterate.proximal_slack);
  }

  const Vector& tolerance() const override { return accepted; }
  const char* name() const override { return "gbcl"; }

  ScheduleSettings settings;
  const Cones* cones = nullptr;
  std::vector<Index> offsets;
  Vector accepted, factor;
  Scalar previous_reference = 0.0;
};

}

std::unique_ptr<Schedule> make_bcl(const ScheduleSettings& settings) {
  return std::make_unique<Bcl>(settings);
}

std::unique_ptr<Schedule> make_interior(const ScheduleSettings& settings) {
  return std::make_unique<Interior>(settings);
}

std::unique_ptr<Schedule> make_gbcl(const ScheduleSettings& settings) {
  return std::make_unique<Gbcl>(settings);
}

std::unique_ptr<Schedule> make_schedule(ScheduleKind kind,
                                        const ScheduleSettings& settings) {
  switch (kind) {
    case ScheduleKind::Interior:
      return make_interior(settings);
    case ScheduleKind::Gbcl:
      return make_gbcl(settings);
    default:
      return make_bcl(settings);
  }
}

}
