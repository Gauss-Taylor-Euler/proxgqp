#include <cstdio>

#include "gblock.hpp"
#include "test/harness.hpp"

#define TEST_NAME "gblock"

namespace proxgqp {
namespace {

GBlock diagonal_block(Index length) {
  GBlock block;
  block.kind = GBlock::Kind::Diagonal;
  block.diagonal.resize(length);
  for (Index i = 0; i < length; ++i) block.diagonal(i) = 1.0 + i;
  return block;
}

GBlock low_rank_block(Index length, Index rank) {
  GBlock block;
  block.kind = GBlock::Kind::DiagPlusLowRank;
  block.delta = 2.0;
  block.low_rank_columns.setZero(length, rank);
  for (Index i = 0; i < length; ++i)
    for (Index j = 0; j < rank; ++j)
      block.low_rank_columns(i, j) = std::cos(1.0 + i + 3.0 * j);
  block.low_rank_middle = DenseMatrix::Identity(rank, rank) * 0.5;
  return block;
}

void block_rows_matches_the_shape() {
  CHECK(block_rows(diagonal_block(5)) == 5);
  CHECK(block_rows(low_rank_block(6, 2)) == 6);
}

void inverting_a_diagonal_block_round_trips() {
  const GBlock block = diagonal_block(5);
  GBlock inverse;
  invert_block(block, inverse);
  Vector v = Vector::LinSpaced(5, 1.0, 2.0);
  Vector image(5), back(5);
  apply_block(block, v, image);
  apply_block(inverse, image, back);
  for (Index i = 0; i < 5; ++i) CHECK_CLOSE(v(i), back(i), 1e-10);
}

void inverting_a_low_rank_block_round_trips() {
  const GBlock block = low_rank_block(6, 2);
  GBlock inverse;
  invert_block(block, inverse);
  Vector v = Vector::LinSpaced(6, -1.0, 1.0);
  Vector image(6), back(6);
  apply_block(block, v, image);
  apply_block(inverse, image, back);
  for (Index i = 0; i < 6; ++i) CHECK_CLOSE(v(i), back(i), 1e-9);
}

void the_penalty_lands_on_the_diagonal() {
  GBlock block = diagonal_block(4);
  const Vector before = block.diagonal;
  Vector penalty = Vector::Constant(4, 0.25);
  add_penalty(penalty, block);
  for (Index i = 0; i < 4; ++i)
    CHECK_CLOSE(block.diagonal(i), before(i) + 0.25, 1e-12);
}

void a_block_is_symmetric() {
  const GBlock block = low_rank_block(5, 2);
  Vector u = Vector::LinSpaced(5, 0.3, 1.7);
  Vector v = Vector::LinSpaced(5, -1.1, 0.9);
  Vector image_u(5), image_v(5);
  apply_block(block, u, image_u);
  apply_block(block, v, image_v);
  CHECK_CLOSE(u.dot(image_v), v.dot(image_u), 1e-10);
}

void run_all() {
  RUN(block_rows_matches_the_shape);
  RUN(inverting_a_diagonal_block_round_trips);
  RUN(inverting_a_low_rank_block_round_trips);
  RUN(the_penalty_lands_on_the_diagonal);
  RUN(a_block_is_symmetric);
}

}
}

using proxgqp::run_all;
TEST_MAIN
