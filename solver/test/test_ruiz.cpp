#include <cstdio>
#include <vector>

#include "ruiz/ruiz.hpp"
#include "test/harness.hpp"

#define TEST_NAME "ruiz"

namespace proxgqp {
namespace {

SparseMatrix dense_from(const DenseMatrix& given) {
  return given.sparseView();
}

void an_empty_row_is_left_alone() {
  const Index columns = 3, rows = 4;
  DenseMatrix constraint = DenseMatrix::Zero(rows, columns);
  constraint(0, 0) = 2.0;
  constraint(1, 1) = 4.0;
  constraint(3, 2) = 8.0;
  SparseMatrix E = dense_from(constraint);
  SparseMatrix P = dense_from(DenseMatrix::Identity(columns, columns));
  Vector q = Vector::Ones(columns);
  Vector f = Vector::Ones(rows);
  const Cones cones{Nonneg{static_cast<std::size_t>(rows)}};

  ScaledProblem scaled;
  RuizSettings settings;
  equilibrate(P, q, E, f, cones, settings, scaled);

  CHECK_CLOSE(scaled.row_scale(2), 1.0, 1e-12);
  for (Index row = 0; row < rows; ++row) {
    CHECK(std::isfinite(scaled.row_scale(row)));
    CHECK(scaled.row_scale(row) < 1e6);
    CHECK(scaled.row_scale(row) > 1e-6);
  }
}

void an_empty_column_is_left_alone() {
  const Index columns = 3, rows = 2;
  DenseMatrix constraint = DenseMatrix::Zero(rows, columns);
  constraint(0, 0) = 3.0;
  constraint(1, 2) = 5.0;
  SparseMatrix E = dense_from(constraint);
  DenseMatrix objective = DenseMatrix::Zero(columns, columns);
  objective(0, 0) = 1.0;
  objective(2, 2) = 1.0;
  SparseMatrix P = dense_from(objective);
  Vector q = Vector::Zero(columns);
  Vector f = Vector::Ones(rows);
  const Cones cones{Nonneg{static_cast<std::size_t>(rows)}};

  ScaledProblem scaled;
  RuizSettings settings;
  equilibrate(P, q, E, f, cones, settings, scaled);
  CHECK_CLOSE(scaled.column_scale(1), 1.0, 1e-12);
}

void the_scaling_round_trips() {
  const Index columns = 4, rows = 3;
  DenseMatrix constraint(rows, columns);
  for (Index i = 0; i < rows; ++i)
    for (Index j = 0; j < columns; ++j)
      constraint(i, j) = std::sin(1.0 + i * 3.0 + j);
  SparseMatrix E = dense_from(constraint);
  DenseMatrix objective = DenseMatrix::Identity(columns, columns) * 7.0;
  SparseMatrix P = dense_from(objective);
  Vector q = Vector::LinSpaced(columns, 1.0, 4.0);
  Vector f = Vector::LinSpaced(rows, 1.0, 3.0);
  const Cones cones{Nonneg{static_cast<std::size_t>(rows)}};

  ScaledProblem scaled;
  RuizSettings settings;
  equilibrate(P, q, E, f, cones, settings, scaled);

  Vector point = Vector::LinSpaced(columns, -1.0, 2.0);
  Vector carried(columns), back(columns);
  scale_primal(scaled, point, carried);
  unscale_primal(scaled, carried, back);
  for (Index i = 0; i < columns; ++i) CHECK_CLOSE(point(i), back(i), 1e-12);

  Vector dual = Vector::LinSpaced(rows, 0.5, 1.5);
  Vector carried_dual(rows), back_dual(rows);
  scale_dual(scaled, dual, carried_dual);
  unscale_dual(scaled, carried_dual, back_dual);
  for (Index i = 0; i < rows; ++i) CHECK_CLOSE(dual(i), back_dual(i), 1e-12);
}

void a_curved_block_keeps_one_factor() {
  const Index columns = 3;
  const std::size_t block = 4;
  DenseMatrix constraint(block, columns);
  for (Index i = 0; i < static_cast<Index>(block); ++i)
    for (Index j = 0; j < columns; ++j)
      constraint(i, j) = 1.0 + i * 10.0 + j;
  SparseMatrix E = dense_from(constraint);
  SparseMatrix P = dense_from(DenseMatrix::Identity(columns, columns));
  Vector q = Vector::Ones(columns);
  Vector f = Vector::Ones(block);
  const Cones cones{SecondOrder{block}};

  ScaledProblem scaled;
  RuizSettings settings;
  equilibrate(P, q, E, f, cones, settings, scaled);
  for (Index row = 1; row < static_cast<Index>(block); ++row)
    CHECK_CLOSE(scaled.row_scale(row), scaled.row_scale(0), 1e-12);
}

void run_all() {
  RUN(an_empty_row_is_left_alone);
  RUN(an_empty_column_is_left_alone);
  RUN(the_scaling_round_trips);
  RUN(a_curved_block_keeps_one_factor);
}

}
}

using proxgqp::run_all;
TEST_MAIN
