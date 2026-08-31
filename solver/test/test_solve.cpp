#include <cstdio>

#include "solve.hpp"
#include "test/harness.hpp"

#define TEST_NAME "solve"

namespace proxgqp {
namespace {

SparseMatrix from_dense(const DenseMatrix& given) { return given.sparseView(); }

Results run(const SparseMatrix& P, const Vector& q, const SparseMatrix& E,
            const Vector& f, const Cones& cones, Method method) {
  Settings settings;
  settings.method = method;
  return solve(P, q, E, f, cones, settings, nullptr);
}

void an_unconstrained_quadratic_reaches_its_minimum() {
  const Index columns = 3;
  SparseMatrix P = from_dense(DenseMatrix::Identity(columns, columns) * 2.0);
  Vector q(columns);
  q << -2.0, 4.0, -6.0;
  SparseMatrix E(0, columns);
  Vector f(0);
  const Cones cones{};
  const Results results = run(P, q, E, f, cones, Method::Interior);
  CHECK(results.status == Status::Solved);
  for (Index i = 0; i < columns; ++i)
    CHECK_CLOSE(results.x(i), -q(i) / 2.0, 1e-6);
}

void a_problem_with_no_rows_does_not_crash() {
  const Index columns = 3;
  SparseMatrix P = from_dense(DenseMatrix::Identity(columns, columns) * 2.0);
  Vector q(columns);
  q << -2.0, 4.0, -6.0;
  SparseMatrix E(0, columns);
  Vector f(0);
  const Cones cones{};
  const Results results = run(P, q, E, f, cones, Method::Semismooth);
  CHECK(results.x.allFinite());
  for (Index i = 0; i < columns; ++i)
    CHECK_CLOSE(results.x(i), -q(i) / 2.0, 1e-4);
}

void an_equality_pins_the_solution() {
  const Index columns = 2;
  SparseMatrix P = from_dense(DenseMatrix::Identity(columns, columns) * 2.0);
  Vector q = Vector::Zero(columns);
  DenseMatrix constraint(1, columns);
  constraint << 1.0, 1.0;
  SparseMatrix E = from_dense(constraint);
  Vector f(1);
  f << 2.0;
  const Cones cones{Zero{1}};
  for (Method smoothing :
       {Method::Interior, Method::Semismooth}) {
    const Results results = run(P, q, E, f, cones, smoothing);
    CHECK(results.status == Status::Solved);
    CHECK_CLOSE(results.x(0), 1.0, 1e-6);
    CHECK_CLOSE(results.x(1), 1.0, 1e-6);
  }
}

void a_bound_becomes_active() {
  const Index columns = 1;
  SparseMatrix P = from_dense(DenseMatrix::Identity(columns, columns) * 2.0);
  Vector q(columns);
  q << 4.0;
  DenseMatrix constraint(1, columns);
  constraint << -1.0;
  SparseMatrix E = from_dense(constraint);
  Vector f(1);
  f << 0.0;
  const Cones cones{Nonneg{1}};
  for (Method smoothing :
       {Method::Interior, Method::Semismooth}) {
    const Results results = run(P, q, E, f, cones, smoothing);
    CHECK(results.status == Status::Solved);
    CHECK_CLOSE(results.x(0), 0.0, 1e-6);
  }
}

void the_returned_slack_satisfies_the_splitting() {
  const Index columns = 2;
  SparseMatrix P = from_dense(DenseMatrix::Identity(columns, columns) * 2.0);
  Vector q(columns);
  q << 1.0, -3.0;
  DenseMatrix constraint(2, columns);
  constraint << -1.0, 0.0, 0.0, -1.0;
  SparseMatrix E = from_dense(constraint);
  Vector f = Vector::Zero(2);
  const Cones cones{Nonneg{2}};
  const Results results = run(P, q, E, f, cones, Method::Interior);
  Vector image = E * results.x;
  for (Index i = 0; i < 2; ++i)
    CHECK_CLOSE(image(i) + results.s(i), f(i), 1e-9);
}

void the_residuals_are_small_at_the_answer() {
  const Index columns = 3;
  DenseMatrix objective = DenseMatrix::Identity(columns, columns) * 3.0;
  SparseMatrix P = from_dense(objective);
  Vector q = Vector::LinSpaced(columns, -2.0, 2.0);
  DenseMatrix constraint = DenseMatrix::Identity(columns, columns) * -1.0;
  SparseMatrix E = from_dense(constraint);
  Vector f = Vector::Zero(columns);
  const Cones cones{Nonneg{static_cast<std::size_t>(columns)}};
  const Results results = run(P, q, E, f, cones, Method::Interior);
  CHECK(results.status == Status::Solved);
  CHECK(results.kkt.primal_residual < 1e-7);
  CHECK(results.kkt.dual_residual < 1e-7);
  CHECK(results.kkt.complementarity < 1e-7);
}

void run_all() {
  RUN(an_unconstrained_quadratic_reaches_its_minimum);
  RUN(a_problem_with_no_rows_does_not_crash);
  RUN(an_equality_pins_the_solution);
  RUN(a_bound_becomes_active);
  RUN(the_returned_slack_satisfies_the_splitting);
  RUN(the_residuals_are_small_at_the_answer);
}

}
}

using proxgqp::run_all;
TEST_MAIN
