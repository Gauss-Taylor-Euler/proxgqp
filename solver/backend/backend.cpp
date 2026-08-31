#include <variant>

#include "backend.hpp"
#include "cone/kernel.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace proxgqp {

namespace {

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::invalid_argument(std::string("pattern: ") + message);
  }
}

}

Index StoredPattern::value_count() const {
  return column_start.empty() ? 0 : Index(column_start.back());
}

void validate_pattern(const Pattern& pattern) {
  require(pattern.dimension >= 0, "negative dimension");
  require(pattern.dimension == 0 || pattern.column_start != nullptr,
          "null column_start");
  if (pattern.dimension == 0) {
    return;
  }
  require(pattern.column_start[0] == 0, "column_start does not begin at zero");
  for (Index column = 0; column < pattern.dimension; ++column) {
    const StorageIndex begin = pattern.column_start[column];
    const StorageIndex end = pattern.column_start[column + 1];
    require(end >= begin, "column_start decreases");
    require(end == begin || pattern.row_index != nullptr, "null row_index");
    require(end > begin, "column has no entries, diagonal is missing");
    require(pattern.row_index[begin] == StorageIndex(column),
            "column does not begin with its diagonal");
    for (StorageIndex position = begin; position < end; ++position) {
      const StorageIndex row = pattern.row_index[position];
      require(row >= StorageIndex(column), "entry above the diagonal");
      require(row < StorageIndex(pattern.dimension), "row index out of range");
      require(position == begin || row > pattern.row_index[position - 1],
              "rows are not sorted within a column");
    }
  }
}

StoredPattern copy_pattern(const Pattern& pattern) {
  validate_pattern(pattern);
  StoredPattern stored;
  stored.dimension = pattern.dimension;
  stored.column_start.assign(pattern.column_start,
                             pattern.column_start + pattern.dimension + 1);
  const Index count = stored.value_count();
  stored.row_index.assign(pattern.row_index, pattern.row_index + count);
  return stored;
}

SparseMatrix lower_matrix_from(const StoredPattern& stored) {
  SparseMatrix matrix(stored.dimension, stored.dimension);
  const Index count = stored.value_count();
  matrix.resizeNonZeros(count);
  std::copy(stored.column_start.begin(), stored.column_start.end(),
            matrix.outerIndexPtr());
  std::copy(stored.row_index.begin(), stored.row_index.end(),
            matrix.innerIndexPtr());
  std::fill(matrix.valuePtr(), matrix.valuePtr() + count, Scalar(0));
  return matrix;
}

BackendKind resolve_backend(BackendKind requested, const Cones& cones) {
  if (requested != BackendKind::Auto) return requested;
  (void)cones;
  return BackendKind::Cholmod;
}

std::unique_ptr<Backend> make_backend(BackendKind kind, bool positive_definite) {
  switch (kind) {
    case BackendKind::EigenSparse:
      return make_eigen_sparse(positive_definite);
    case BackendKind::EigenDense:
      return make_eigen_dense(positive_definite);
    case BackendKind::Qdldl:
      return make_qdldl(positive_definite);
    case BackendKind::Cholmod:
      return make_cholmod(positive_definite);
    case BackendKind::LapackDense:
      return make_lapack_dense(positive_definite);
    case BackendKind::Auto:
      break;
  }
  throw std::invalid_argument("make_backend: unknown backend kind");
}

}
