#include <cstdint>
#include <cstring>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include "solve.hpp"

namespace nb = nanobind;
using namespace proxgqp;

namespace {

using DoubleArray = nb::ndarray<const double, nb::ndim<1>, nb::c_contig>;
using IndexArray = nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig>;
using SizeArray = nb::ndarray<const std::int64_t, nb::ndim<1>, nb::c_contig>;
using Output = nb::ndarray<nb::numpy, double, nb::ndim<1>>;

Output owned(Vector&& source) {
  auto* held = new Vector(std::move(source));
  nb::capsule owner(held, [](void* pointer) noexcept {
    delete static_cast<Vector*>(pointer);
  });
  const std::size_t length = static_cast<std::size_t>(held->size());
  return Output(held->data(), {length}, owner);
}

SparseMatrix from_compressed(Index rows, Index columns, const IndexArray& start,
                             const IndexArray& row, const DoubleArray& value) {
  SparseMatrix matrix(rows, columns);
  const StorageIndex* starts = start.data();
  const Index entries = starts[columns];
  matrix.resizeNonZeros(entries);
  std::memcpy(matrix.outerIndexPtr(), starts,
              sizeof(StorageIndex) * static_cast<std::size_t>(columns + 1));
  if (entries > 0) {
    std::memcpy(matrix.innerIndexPtr(), row.data(),
                sizeof(StorageIndex) * static_cast<std::size_t>(entries));
    std::memcpy(matrix.valuePtr(), value.data(),
                sizeof(Scalar) * static_cast<std::size_t>(entries));
  }
  return matrix;
}

Cones build_cones(const IndexArray& kind, const SizeArray& size,
                  const DoubleArray& exponent) {
  Cones cones;
  const std::size_t count = kind.shape(0);
  cones.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    const std::size_t rows = static_cast<std::size_t>(size(index));
    switch (kind(index)) {
      case 0: cones.push_back(Zero{rows}); break;
      case 1: cones.push_back(Nonneg{rows}); break;
      case 2: cones.push_back(SecondOrder{rows}); break;
      case 3: cones.push_back(PSDTriangle::of_side(rows)); break;
      case 4: cones.push_back(Exponential{}); break;
      default: cones.push_back(Power{exponent(index), 3}); break;
    }
  }
  return cones;
}

}

NB_MODULE(_core, handle) {
  nb::enum_<Status>(handle, "Status")
      .value("solved", Status::Solved)
      .value("primal_infeasible", Status::PrimalInfeasible)
      .value("dual_infeasible", Status::DualInfeasible)
      .value("max_iter_outer", Status::MaxIterOuter)
      .value("max_newton", Status::MaxNewton)
      .value("numerical_failure", Status::NumericalFailure);

  nb::enum_<Method>(handle, "Method")
      .value("semismooth", Method::Semismooth)
      .value("interior", Method::Interior)
      .value("interior_exp", Method::InteriorExp);

  nb::enum_<LineSearchKind>(handle, "LineSearch")
      .value("exact", LineSearchKind::Exact)
      .value("armijo", LineSearchKind::Armijo)
      .value("decrease", LineSearchKind::Decrease);

  nb::enum_<Penalty>(handle, "Penalty")
      .value("bcl", Penalty::Bcl)
      .value("gbcl", Penalty::GBcl)
      .value("gbcl_exp", Penalty::GBclExp);

  nb::enum_<RoadKind>(handle, "Road")
      .value("automatic", RoadKind::Auto)
      .value("three_by_three", RoadKind::ThreeByThree)
      .value("schur", RoadKind::Schur);

  nb::enum_<BackendKind>(handle, "Backend")
      .value("eigen_sparse", BackendKind::EigenSparse)
      .value("eigen_dense", BackendKind::EigenDense)
      .value("qdldl", BackendKind::Qdldl)
      .value("cholmod", BackendKind::Cholmod)
      .value("lapack_dense", BackendKind::LapackDense)
      .value("automatic", BackendKind::Auto);

#define PROXGQP_SHARED(CLASS)                                                 \
  .def_rw("rho", &CLASS::rho)                                                 \
      .def_rw("rho_p", &CLASS::rho_p)                                         \
      .def_rw("mu_eq", &CLASS::mu_eq)                                         \
      .def_rw("rho_p_decay", &CLASS::rho_p_decay)                             \
      .def_rw("rho_p_min", &CLASS::rho_p_min)                                 \
      .def_rw("refactor_bump", &CLASS::refactor_bump)                         \
      .def_rw("max_refactor", &CLASS::max_refactor)                           \
      .def_rw("ruiz_iter", &CLASS::ruiz_iter)                                 \
      .def_rw("refine", &CLASS::refine)                                       \
      .def_rw("infeas_repeat", &CLASS::infeas_repeat)                         \
      .def_rw("eps_infeas", &CLASS::eps_infeas)                               \
      .def_rw("stall_tol", &CLASS::stall_tol)

  nb::class_<Tuning::Semismooth>(handle, "Semismooth")
      .def(nb::init<>())
      PROXGQP_SHARED(Tuning::Semismooth)
      .def_rw("mu_in", &Tuning::Semismooth::mu_in)
      .def_rw("line_search", &Tuning::Semismooth::line_search)
      .def_rw("penalty", &Tuning::Semismooth::penalty)
      .def_rw("ls_sigma", &Tuning::Semismooth::ls_sigma)
      .def_rw("ls_beta", &Tuning::Semismooth::ls_beta)
      .def_rw("ls_max_back", &Tuning::Semismooth::ls_max_back)
      .def_rw("ls_max_secant", &Tuning::Semismooth::ls_max_secant);

  nb::class_<Tuning::Bcl>(handle, "Bcl")
      .def(nb::init<>())
      .def_rw("mu_in", &Tuning::Bcl::mu_in)
      .def_rw("mu_eq", &Tuning::Bcl::mu_eq)
      .def_rw("alpha", &Tuning::Bcl::alpha)
      .def_rw("beta", &Tuning::Bcl::beta)
      .def_rw("mu_update_factor", &Tuning::Bcl::mu_update_factor)
      .def_rw("mu_min_in", &Tuning::Bcl::mu_min_in)
      .def_rw("mu_min_eq", &Tuning::Bcl::mu_min_eq)
      .def_rw("cold_reset", &Tuning::Bcl::cold_reset)
      .def_rw("cold_reset_threshold", &Tuning::Bcl::cold_reset_threshold)
      .def_rw("rho_p_outer", &Tuning::Bcl::rho_p_outer)
      .def_rw("safe_guard", &Tuning::Bcl::safe_guard)
      .def_rw("revert_on_reject", &Tuning::Bcl::revert_on_reject);

  nb::class_<Tuning::Gbcl>(handle, "Gbcl")
      .def(nb::init<>())
      .def_rw("mu_in", &Tuning::Gbcl::mu_in)
      .def_rw("mu_eq", &Tuning::Gbcl::mu_eq)
      .def_rw("mu_exp", &Tuning::Gbcl::mu_exp)
      .def_rw("penalty_reduction", &Tuning::Gbcl::penalty_reduction)
      .def_rw("mu_min", &Tuning::Gbcl::mu_min)
      .def_rw("mu_cut", &Tuning::Gbcl::mu_cut)
      .def_rw("alpha", &Tuning::Gbcl::alpha)
      .def_rw("beta", &Tuning::Gbcl::beta)
      .def_rw("eps_outer_init", &Tuning::Gbcl::eps_outer_init)
      .def_rw("smallest_tolerance", &Tuning::Gbcl::smallest_tolerance)
      .def_rw("rho_p_outer", &Tuning::Gbcl::rho_p_outer);

  nb::class_<Tuning::GbclExp>(handle, "GbclExp")
      .def(nb::init<>())
      .def_rw("mu_in", &Tuning::GbclExp::mu_in)
      .def_rw("mu_eq", &Tuning::GbclExp::mu_eq)
      .def_rw("alpha", &Tuning::GbclExp::alpha)
      .def_rw("beta", &Tuning::GbclExp::beta)
      .def_rw("mu_exp", &Tuning::GbclExp::mu_exp)
      .def_rw("penalty_reduction", &Tuning::GbclExp::penalty_reduction)
      .def_rw("mu_adapt", &Tuning::GbclExp::mu_adapt)
      .def_rw("mu_min", &Tuning::GbclExp::mu_min)
      .def_rw("eps_outer_init", &Tuning::GbclExp::eps_outer_init)
      .def_rw("eps_newton_init", &Tuning::GbclExp::eps_newton_init)
      .def_rw("rho_p_outer", &Tuning::GbclExp::rho_p_outer)
      .def_rw("safe_guard", &Tuning::GbclExp::safe_guard);

  nb::class_<Tuning::Interior>(handle, "Interior")
      .def(nb::init<>())
      PROXGQP_SHARED(Tuning::Interior)
      .def_rw("tau", &Tuning::Interior::tau)
      .def_rw("tau_curved", &Tuning::Interior::tau_curved)
      .def_rw("mehrotra", &Tuning::Interior::mehrotra)
      .def_rw("sigma_centre", &Tuning::Interior::sigma_centre)
      .def_rw("sigma_min", &Tuning::Interior::sigma_min)
      .def_rw("sigma_max", &Tuning::Interior::sigma_max)
      .def_rw("eps_reg", &Tuning::Interior::eps_reg)
      .def_rw("seeded_start", &Tuning::Interior::seeded_start)
      .def_rw("pad_constant", &Tuning::Interior::pad_constant)
      .def_rw("pad_fraction", &Tuning::Interior::pad_fraction)
      .def_rw("reg_floor", &Tuning::Interior::reg_floor)
      .def_rw("reg_fine", &Tuning::Interior::reg_fine)
      .def_rw("improvement", &Tuning::Interior::improvement)
      .def_rw("stalled_rate", &Tuning::Interior::stalled_rate)
      .def_rw("stall_before_fine", &Tuning::Interior::stall_before_fine)
      .def_rw("early_outers", &Tuning::Interior::early_outers)
      .def_rw("degenerate_step", &Tuning::Interior::degenerate_step)
      .def_rw("refactor_ratchet", &Tuning::Interior::refactor_ratchet);


#undef PROXGQP_SHARED

  nb::class_<Tuning>(handle, "Tuning")
      .def(nb::init<>())
      .def_rw("semismooth", &Tuning::semismooth)
      .def_rw("bcl", &Tuning::bcl)
      .def_rw("gbcl", &Tuning::gbcl)
      .def_rw("gbcl_exp", &Tuning::gbcl_exp)
      .def_rw("interior", &Tuning::interior)
      .def_rw("interior_exp", &Tuning::interior_exp);

  nb::class_<Settings>(handle, "Settings")
      .def(nb::init<>())
      .def_rw("method", &Settings::method)
      .def_rw("eps_abs", &Settings::eps_abs)
      .def_rw("eps_rel", &Settings::eps_rel)
      .def_rw("eps_gap_abs", &Settings::eps_gap_abs)
      .def_rw("eps_gap_rel", &Settings::eps_gap_rel)
      .def_rw("max_iter_outer", &Settings::max_iter_outer)
      .def_rw("max_iter_inner", &Settings::max_iter_inner)
      .def_rw("max_newton", &Settings::max_newton)
      .def_rw("road", &Settings::road)
      .def_rw("backend", &Settings::backend)
      .def_rw("equilibrate", &Settings::equilibrate)
      .def_rw("scale_cost", &Settings::scale_cost)
      .def_rw("max_threads", &Settings::max_threads)
      .def_rw("verbose", &Settings::verbose)
      .def_rw("tuning", &Settings::tuning);

  nb::class_<Results>(handle, "Results")
      .def_ro("status", &Results::status)
      .def_ro("outer_iterations", &Results::outer_iterations)
      .def_ro("inner_iterations", &Results::inner_iterations)
      .def_ro("seconds", &Results::seconds)
      .def_prop_ro("x", [](Results& r) { return owned(Vector(r.x)); },
                   nb::rv_policy::reference)
      .def_prop_ro("s", [](Results& r) { return owned(Vector(r.s)); },
                   nb::rv_policy::reference)
      .def_prop_ro("z", [](Results& r) { return owned(Vector(r.z)); },
                   nb::rv_policy::reference)
      .def_prop_ro("primal_residual", [](const Results& r) { return r.kkt.primal_residual; })
      .def_prop_ro("dual_residual", [](const Results& r) { return r.kkt.dual_residual; })
      .def_prop_ro("relative_primal_residual",
                   [](const Results& r) { return r.kkt.relative_primal_residual; })
      .def_prop_ro("relative_dual_residual",
                   [](const Results& r) { return r.kkt.relative_dual_residual; })
      .def_prop_ro("dual_cone_violation", [](const Results& r) { return r.kkt.dual_cone_violation; })
      .def_prop_ro("complementarity", [](const Results& r) { return r.kkt.complementarity; })
      .def_prop_ro("objective", [](const Results& r) { return r.kkt.objective; });

  handle.def(
      "solve",
      [](Index columns, Index rows, IndexArray objective_start,
         IndexArray objective_row, DoubleArray objective_value,
         DoubleArray linear, IndexArray constraint_start,
         IndexArray constraint_row, DoubleArray constraint_value,
         DoubleArray offset, IndexArray cone_kind, SizeArray cone_size,
         DoubleArray cone_exponent, const Settings& settings,
         nb::object warm) {
        const SparseMatrix P = from_compressed(columns, columns, objective_start,
                                               objective_row, objective_value);
        const SparseMatrix E = from_compressed(rows, columns, constraint_start,
                                               constraint_row, constraint_value);
        const Vector q = Eigen::Map<const Vector>(linear.data(), columns);
        const Vector b = Eigen::Map<const Vector>(offset.data(), rows);
        const Cones cones = build_cones(cone_kind, cone_size, cone_exponent);

        Results start;
        const Results* warm_pointer = nullptr;
        if (!warm.is_none()) {
          auto triple = nb::cast<nb::tuple>(warm);
          auto warm_x = nb::cast<DoubleArray>(triple[0]);
          auto warm_s = nb::cast<DoubleArray>(triple[1]);
          auto warm_z = nb::cast<DoubleArray>(triple[2]);
          start.x = Eigen::Map<const Vector>(warm_x.data(), columns);
          start.s = Eigen::Map<const Vector>(warm_s.data(), rows);
          start.z = Eigen::Map<const Vector>(warm_z.data(), rows);
          start.has_point = true;
          warm_pointer = &start;
        }
        return solve(P, q, E, b, cones, settings, warm_pointer);
      },
      nb::arg("columns"), nb::arg("rows"), nb::arg("objective_start"),
      nb::arg("objective_row"), nb::arg("objective_value"), nb::arg("linear"),
      nb::arg("constraint_start"), nb::arg("constraint_row"),
      nb::arg("constraint_value"), nb::arg("offset"), nb::arg("cone_kind"),
      nb::arg("cone_size"), nb::arg("cone_exponent"), nb::arg("settings"),
      nb::arg("warm").none() = nb::none());
}
