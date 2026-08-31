#include <algorithm>
#include <memory>

#include <Eigen/SparseCholesky>

#include "backend.hpp"

namespace proxgqp {

namespace {

struct EigenSparse final : Backend {
  StoredPattern stored;
  SparseMatrix matrix;
  Eigen::SimplicialLDLT<SparseMatrix, Eigen::Lower> factorisation;
  bool analysed = false;
  bool factorised = false;

  Capabilities capabilities() const override {
    Capabilities capabilities;
    capabilities.supports_update = false;
    capabilities.max_threads = 1;
    return capabilities;
  }

  void symbolic(const Pattern& pattern) override {
    stored = copy_pattern(pattern);
    matrix = lower_matrix_from(stored);
    factorisation.analyzePattern(matrix);
    analysed = factorisation.info() == Eigen::Success;
    factorised = false;
  }

  Index value_count() const override { return stored.value_count(); }

  bool numeric(const Scalar* values) override {
    if (!analysed) {
      return false;
    }
    std::copy(values, values + stored.value_count(), matrix.valuePtr());
    factorisation.factorize(matrix);
    factorised = factorisation.info() == Eigen::Success;
    return factorised;
  }

  bool update(const Delta&) override { return false; }

  void solve(const Scalar* right_hand_side, Scalar* solution) const override {
    Eigen::Map<const Vector> input(right_hand_side, stored.dimension);
    Eigen::Map<Vector> output(solution, stored.dimension);
    output = factorisation.solve(input);
  }

  void set_threads(std::size_t) override {}
  std::size_t threads() const override { return 1; }
  const char* name() const override { return "eigen_sparse"; }
};

}

std::unique_ptr<Backend> make_eigen_sparse(bool) {
  return std::make_unique<EigenSparse>();
}

}
