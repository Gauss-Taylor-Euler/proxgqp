#include <algorithm>
#include <memory>

#include <Eigen/Cholesky>

#include "backend.hpp"
#include "backend/threads.hpp"

namespace proxgqp {

namespace {

struct EigenDense final : Backend {
  StoredPattern stored;
  DenseMatrix matrix;
  Eigen::LDLT<DenseMatrix, Eigen::Lower> factorisation;
  bool factorised = false;

  Capabilities capabilities() const override {
    Capabilities capabilities;
    capabilities.supports_update = true;
    capabilities.max_threads = available_dense_threads();
    return capabilities;
  }

  void symbolic(const Pattern& pattern) override {
    stored = copy_pattern(pattern);
    matrix.setZero(stored.dimension, stored.dimension);
    factorised = false;
  }

  Index value_count() const override { return stored.value_count(); }

  bool numeric(const Scalar* values) override {
    matrix.setZero();
    for (Index column = 0; column < stored.dimension; ++column) {
      const StorageIndex begin = stored.column_start[column];
      const StorageIndex end = stored.column_start[column + 1];
      for (StorageIndex position = begin; position < end; ++position) {
        const Index row = stored.row_index[position];
        matrix(row, column) = values[position];
        matrix(column, row) = values[position];
      }
    }
    factorisation.compute(matrix);
    factorised = factorisation.info() == Eigen::Success;
    return factorised;
  }

  bool update(const Delta& delta) override {
    if (!factorised || delta.columns == nullptr) {
      return false;
    }
    if (delta.columns->rows() != stored.dimension) {
      return false;
    }
    const Scalar sign = delta.sign >= 0 ? Scalar(1) : Scalar(-1);
    for (Index column = 0; column < delta.columns->cols(); ++column) {
      const Vector dense_column = Vector(delta.columns->col(column));
      factorisation.rankUpdate(dense_column, sign);
      if (factorisation.info() != Eigen::Success) {
        factorised = false;
        return false;
      }
    }
    return true;
  }

  void solve(const Scalar* right_hand_side, Scalar* solution) const override {
    Eigen::Map<const Vector> input(right_hand_side, stored.dimension);
    Eigen::Map<Vector> output(solution, stored.dimension);
    output = factorisation.solve(input);
  }

  void set_threads(std::size_t requested) override {
    active_threads = configure_dense_threads(requested);
  }
  std::size_t threads() const override { return active_threads; }
  std::size_t active_threads = 1;

  const char* name() const override { return "eigen_dense"; }
};

}

std::unique_ptr<Backend> make_eigen_dense(bool) {
  return std::make_unique<EigenDense>();
}

}
