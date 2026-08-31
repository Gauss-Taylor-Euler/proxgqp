#pragma once

#include <cstddef>
#include <vector>

#include "types.hpp"

namespace proxgqp {

struct GBlock {
  enum class Kind { Diagonal, DiagPlusLowRank, Dense };

  Kind kind = Kind::Diagonal;
  Vector diagonal;
  Scalar delta = 0.0;
  DenseMatrix low_rank_columns;
  DenseMatrix low_rank_middle;
  DenseMatrix dense;
};

Index block_rows(const GBlock& block);

void apply_block(const GBlock& block, const ConstVectorRef& v, VectorRef out);

void apply_all(const std::vector<GBlock>& blocks,
               const std::vector<Index>& offsets, const ConstVectorRef& v,
               Vector& out);

void invert_block(const GBlock& block, GBlock& out);

void add_penalty(const ConstVectorRef& penalty, GBlock& block);

void schur_weight(const GBlock& block, const ConstVectorRef& equality_penalty,
                  GBlock& out);

}
