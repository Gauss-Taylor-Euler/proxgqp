#include <algorithm>
#include <memory>
#include <vector>

#include "backend.hpp"

extern "C" {
#include "qdldl.h"
}

namespace proxgqp {

namespace {

struct Qdldl final : Backend {
  StoredPattern stored;
  std::vector<StorageIndex> upper_column_start;
  std::vector<StorageIndex> upper_row_index;
  std::vector<StorageIndex> upper_position_of;
  std::vector<Scalar> upper_values;

  std::vector<StorageIndex> factor_column_start;
  std::vector<StorageIndex> factor_row_index;
  std::vector<Scalar> factor_values;
  std::vector<Scalar> diagonal;
  std::vector<Scalar> inverse_diagonal;
  std::vector<StorageIndex> column_counts;
  std::vector<StorageIndex> tree;
  std::vector<StorageIndex> integer_work;
  std::vector<QDLDL_bool> boolean_work;
  std::vector<Scalar> float_work;
  mutable std::vector<Scalar> scratch;

  bool analysed = false;
  bool factorised = false;

  Capabilities capabilities() const override {
    Capabilities capabilities;
    capabilities.supports_update = false;
    capabilities.max_threads = 1;
    return capabilities;
  }

  void build_upper_from_lower() {
    const Index dimension = stored.dimension;
    const Index count = stored.value_count();
    upper_column_start.assign(dimension + 1, 0);
    for (Index position = 0; position < count; ++position) {
      ++upper_column_start[stored.row_index[position] + 1];
    }
    for (Index column = 0; column < dimension; ++column) {
      upper_column_start[column + 1] += upper_column_start[column];
    }
    upper_row_index.assign(count, 0);
    upper_position_of.assign(count, 0);
    std::vector<StorageIndex> next(upper_column_start.begin(),
                                   upper_column_start.end() - 1);
    for (Index column = 0; column < dimension; ++column) {
      const StorageIndex begin = stored.column_start[column];
      const StorageIndex end = stored.column_start[column + 1];
      for (StorageIndex position = begin; position < end; ++position) {
        const StorageIndex row = stored.row_index[position];
        const StorageIndex target = next[row]++;
        upper_row_index[target] = StorageIndex(column);
        upper_position_of[position] = target;
      }
    }
    upper_values.assign(count, Scalar(0));
  }

  void symbolic(const Pattern& pattern) override {
    stored = copy_pattern(pattern);
    build_upper_from_lower();
    const Index dimension = stored.dimension;
    column_counts.assign(dimension, 0);
    tree.assign(dimension, 0);
    integer_work.assign(3 * dimension, 0);
    boolean_work.assign(dimension, 0);
    float_work.assign(dimension, Scalar(0));
    scratch.assign(dimension, Scalar(0));

    std::vector<StorageIndex> etree_work(dimension, 0);
    const QDLDL_int factor_size =
        QDLDL_etree(QDLDL_int(dimension), upper_column_start.data(),
                    upper_row_index.data(), etree_work.data(),
                    column_counts.data(), tree.data());
    analysed = factor_size >= 0;
    factorised = false;
    if (!analysed) {
      return;
    }
    factor_column_start.assign(dimension + 1, 0);
    factor_row_index.assign(std::size_t(factor_size), 0);
    factor_values.assign(std::size_t(factor_size), Scalar(0));
    diagonal.assign(dimension, Scalar(0));
    inverse_diagonal.assign(dimension, Scalar(0));
  }

  Index value_count() const override { return stored.value_count(); }

  bool numeric(const Scalar* values) override {
    if (!analysed) {
      return false;
    }
    const Index count = stored.value_count();
    for (Index position = 0; position < count; ++position) {
      upper_values[upper_position_of[position]] = values[position];
    }
    const QDLDL_int factored = QDLDL_factor(
        QDLDL_int(stored.dimension), upper_column_start.data(),
        upper_row_index.data(), upper_values.data(), factor_column_start.data(),
        factor_row_index.data(), factor_values.data(), diagonal.data(),
        inverse_diagonal.data(), column_counts.data(), tree.data(),
        boolean_work.data(), integer_work.data(), float_work.data());
    factorised = factored >= 0;
    return factorised;
  }

  bool update(const Delta&) override { return false; }

  void solve(const Scalar* right_hand_side, Scalar* solution) const override {
    std::copy(right_hand_side, right_hand_side + stored.dimension, solution);
    QDLDL_solve(QDLDL_int(stored.dimension), factor_column_start.data(),
                factor_row_index.data(), factor_values.data(),
                inverse_diagonal.data(), solution);
  }

  void set_threads(std::size_t) override {}
  std::size_t threads() const override { return 1; }
  const char* name() const override { return "qdldl"; }
};

}

std::unique_ptr<Backend> make_qdldl(bool) {
  return std::make_unique<Qdldl>();
}

}
