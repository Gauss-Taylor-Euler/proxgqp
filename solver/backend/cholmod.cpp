#include <algorithm>
#include <memory>
#include <stdexcept>

#include <cholmod.h>

#include "backend.hpp"
#include "backend/threads.hpp"

namespace proxgqp {

namespace {

struct Cholmod final : Backend {
  bool positive_definite = true;
  StoredPattern stored;
  std::vector<Scalar> values;
  mutable cholmod_common common;
  cholmod_sparse view{};
  cholmod_factor* factor = nullptr;
  mutable cholmod_dense right_hand_side_view{};
  mutable cholmod_dense* solution_dense = nullptr;
  mutable cholmod_dense* solve_work = nullptr;
  mutable cholmod_dense* solve_scratch = nullptr;
  bool factorised = false;
  std::size_t requested_threads = 1;

  explicit Cholmod(bool definite) : positive_definite(definite) {
    cholmod_start(&common);
    common.supernodal = positive_definite ? CHOLMOD_AUTO : CHOLMOD_SIMPLICIAL;
    common.final_ll = positive_definite ? 1 : 0;
  }

  ~Cholmod() override {
    if (solve_scratch != nullptr) {
      cholmod_free_dense(&solve_scratch, &common);
    }
    if (solve_work != nullptr) {
      cholmod_free_dense(&solve_work, &common);
    }
    if (solution_dense != nullptr) {
      cholmod_free_dense(&solution_dense, &common);
    }
    if (factor != nullptr) {
      cholmod_free_factor(&factor, &common);
    }
    cholmod_finish(&common);
  }

  Capabilities capabilities() const override {
    Capabilities capabilities;
    capabilities.supports_update = false;
    capabilities.max_threads =
        positive_definite ? available_dense_threads() : 1;
    return capabilities;
  }

  void symbolic(const Pattern& pattern) override {
    stored = copy_pattern(pattern);
    values.assign(std::size_t(stored.value_count()), Scalar(0));

    view.nrow = std::size_t(stored.dimension);
    view.ncol = std::size_t(stored.dimension);
    view.nzmax = std::size_t(stored.value_count());
    view.p = stored.column_start.data();
    view.i = stored.row_index.data();
    view.x = values.data();
    view.nz = nullptr;
    view.z = nullptr;
    view.stype = -1;
    view.itype = CHOLMOD_INT;
    view.xtype = CHOLMOD_REAL;
    view.dtype = CHOLMOD_DOUBLE;
    view.sorted = 1;
    view.packed = 1;

    if (factor != nullptr) {
      cholmod_free_factor(&factor, &common);
    }
    factor = cholmod_analyze(&view, &common);
    if (factor == nullptr) {
      throw std::invalid_argument("cholmod: analyze failed");
    }
    factorised = false;
  }

  Index value_count() const override { return stored.value_count(); }

  bool numeric(const Scalar* incoming) override {
    std::copy(incoming, incoming + stored.value_count(), values.begin());
    const int status = cholmod_factorize(&view, factor, &common);
    factorised = status != 0 && common.status == CHOLMOD_OK;
    return factorised;
  }

  bool update(const Delta&) override { return false; }

  void solve(const Scalar* incoming, Scalar* solution) const override {
    right_hand_side_view.nrow = std::size_t(stored.dimension);
    right_hand_side_view.ncol = 1;
    right_hand_side_view.nzmax = std::size_t(stored.dimension);
    right_hand_side_view.d = std::size_t(stored.dimension);
    right_hand_side_view.x = const_cast<Scalar*>(incoming);
    right_hand_side_view.z = nullptr;
    right_hand_side_view.xtype = CHOLMOD_REAL;
    right_hand_side_view.dtype = CHOLMOD_DOUBLE;

    cholmod_solve2(CHOLMOD_A, factor, &right_hand_side_view, nullptr,
                   &solution_dense, nullptr, &solve_work, &solve_scratch,
                   &common);
    const Scalar* produced = static_cast<const Scalar*>(solution_dense->x);
    std::copy(produced, produced + stored.dimension, solution);
  }

  void set_threads(std::size_t requested) override {
    requested_threads =
        positive_definite ? configure_dense_threads(requested) : 1;
  }

  std::size_t threads() const override {
    return positive_definite ? requested_threads : 1;
  }

  const char* name() const override { return "cholmod"; }
};

}

std::unique_ptr<Backend> make_cholmod(bool positive_definite) {
  return std::make_unique<Cholmod>(positive_definite);
}

}
