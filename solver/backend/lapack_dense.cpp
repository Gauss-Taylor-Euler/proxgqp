#include <algorithm>
#include <memory>
#include <vector>

#include "backend.hpp"
#include "backend/threads.hpp"

extern "C" {
void dpotrf_(const char* triangle, const int* dimension, double* matrix,
             const int* leading_dimension, int* status);
void dpotrs_(const char* triangle, const int* dimension, const int* column_count,
             const double* matrix, const int* leading_dimension, double* vector,
             const int* vector_leading_dimension, int* status);
void dsytrf_(const char* triangle, const int* dimension, double* matrix,
             const int* leading_dimension, int* pivots, double* work,
             const int* work_size, int* status);
void dsytrs_(const char* triangle, const int* dimension, const int* column_count,
             const double* matrix, const int* leading_dimension, const int* pivots,
             double* vector, const int* vector_leading_dimension, int* status);
}

namespace proxgqp {

namespace {

constexpr char kLowerTriangle = 'L';

struct LapackDense final : Backend {
  bool positive_definite = false;
  StoredPattern stored;
  std::vector<Scalar> matrix;
  std::vector<int> pivots;
  std::vector<Scalar> work;
  bool factorised = false;

  explicit LapackDense(bool definite) : positive_definite(definite) {}

  Capabilities capabilities() const override {
    Capabilities capabilities;
    capabilities.supports_update = false;
    capabilities.max_threads = available_dense_threads();
    return capabilities;
  }

  void symbolic(const Pattern& pattern) override {
    stored = copy_pattern(pattern);
    const std::size_t dimension = std::size_t(stored.dimension);
    matrix.assign(dimension * dimension, Scalar(0));
    pivots.assign(dimension, 0);
    work.assign(std::max<std::size_t>(dimension * 64, 1), Scalar(0));
    factorised = false;
  }

  Index value_count() const override { return stored.value_count(); }

  bool numeric(const Scalar* values) override {
    const int dimension = int(stored.dimension);
    std::fill(matrix.begin(), matrix.end(), Scalar(0));
    for (Index column = 0; column < stored.dimension; ++column) {
      const StorageIndex begin = stored.column_start[column];
      const StorageIndex end = stored.column_start[column + 1];
      for (StorageIndex position = begin; position < end; ++position) {
        const Index row = stored.row_index[position];
        matrix[std::size_t(column * Index(dimension) + row)] = values[position];
      }
    }
    int status = 0;
    if (positive_definite) {
      dpotrf_(&kLowerTriangle, &dimension, matrix.data(), &dimension, &status);
    } else {
      const int work_size = int(work.size());
      dsytrf_(&kLowerTriangle, &dimension, matrix.data(), &dimension,
              pivots.data(), work.data(), &work_size, &status);
    }
    factorised = status == 0;
    return factorised;
  }

  bool update(const Delta&) override { return false; }

  void solve(const Scalar* right_hand_side, Scalar* solution) const override {
    const int dimension = int(stored.dimension);
    const int column_count = 1;
    int status = 0;
    std::copy(right_hand_side, right_hand_side + stored.dimension, solution);
    if (positive_definite) {
      dpotrs_(&kLowerTriangle, &dimension, &column_count, matrix.data(),
              &dimension, solution, &dimension, &status);
    } else {
      dsytrs_(&kLowerTriangle, &dimension, &column_count, matrix.data(),
              &dimension, pivots.data(), solution, &dimension, &status);
    }
  }

  void set_threads(std::size_t requested) override {
    active_threads = configure_dense_threads(requested);
  }
  std::size_t threads() const override { return active_threads; }
  std::size_t active_threads = 1;

  const char* name() const override { return "lapack_dense"; }
};

}

std::unique_ptr<Backend> make_lapack_dense(bool positive_definite) {
  return std::make_unique<LapackDense>(positive_definite);
}

}
