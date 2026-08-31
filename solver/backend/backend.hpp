#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "cone_types.hpp"
#include "types.hpp"

namespace proxgqp {

struct Capabilities {
  bool supports_update = false;
  std::size_t max_threads = 1;
};

struct Pattern {
  Index dimension = 0;
  const StorageIndex* column_start = nullptr;
  const StorageIndex* row_index = nullptr;
};

struct StoredPattern {
  Index dimension = 0;
  std::vector<StorageIndex> column_start;
  std::vector<StorageIndex> row_index;

  Index value_count() const;
};

struct Delta {
  const SparseMatrix* columns = nullptr;
  int sign = 1;
};

struct Backend {
  virtual ~Backend() = default;

  virtual Capabilities capabilities() const = 0;
  virtual void symbolic(const Pattern& pattern) = 0;
  virtual Index value_count() const = 0;
  virtual bool numeric(const Scalar* values) = 0;
  virtual bool update(const Delta& delta) = 0;
  virtual void solve(const Scalar* right_hand_side, Scalar* solution) const = 0;
  virtual void set_threads(std::size_t requested) = 0;
  virtual std::size_t threads() const = 0;
  virtual const char* name() const = 0;
};

enum class BackendKind {
  EigenSparse,
  EigenDense,
  Qdldl,
  Cholmod,
  LapackDense,
  Auto,
};

void validate_pattern(const Pattern& pattern);
StoredPattern copy_pattern(const Pattern& pattern);
SparseMatrix lower_matrix_from(const StoredPattern& stored);

std::unique_ptr<Backend> make_backend(BackendKind kind, bool positive_definite);
BackendKind resolve_backend(BackendKind requested, const Cones& cones);

std::unique_ptr<Backend> make_eigen_sparse(bool positive_definite);
std::unique_ptr<Backend> make_eigen_dense(bool positive_definite);
std::unique_ptr<Backend> make_qdldl(bool positive_definite);
std::unique_ptr<Backend> make_cholmod(bool positive_definite);
std::unique_ptr<Backend> make_lapack_dense(bool positive_definite);

}
