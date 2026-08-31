#include <cstdio>
#include <vector>

#include "cone/kernel.hpp"
#include "test/harness.hpp"

#define TEST_NAME "cone"

namespace proxgqp {
namespace {

Cones every_cone() {
  return Cones{Zero{3}, Nonneg{4}, SecondOrder{4}, PSDTriangle::of_side(3),
               Exponential{}, Power{0.35, 3}};
}

Vector sample(Index length, Scalar seed) {
  Vector v(length);
  for (Index i = 0; i < length; ++i)
    v(i) = std::sin(seed * (i + 1)) * (1.0 + 0.5 * i);
  return v;
}

void rows_match_the_declared_dimension() {
  for (const Cone& cone : every_cone()) CHECK(rows(cone) == dimension(cone));
  CHECK(rows(PSDTriangle::of_side(3)) == 6);
  CHECK(rows(Exponential{}) == 3);
}

void projection_is_idempotent() {
  for (const Cone& cone : every_cone()) {
    const Index length = rows(cone);
    Vector once(length), twice(length);
    const Vector v = sample(length, 0.7);
    evaluate_projection(cone, v, once);
    evaluate_projection(cone, once, twice);
    for (Index i = 0; i < length; ++i) CHECK_CLOSE(once(i), twice(i), 1e-7);
  }
}

void projection_lands_in_the_cone() {
  for (const Cone& cone : every_cone()) {
    const Index length = rows(cone);
    Vector projected(length);
    evaluate_projection(cone, sample(length, 1.3), projected);
    CHECK(contains(cone, ConeSide::Dual, projected));
  }
}

void the_projection_is_orthogonal_to_the_residual() {
  for (const Cone& cone : every_cone()) {
    const Index length = rows(cone);
    const Vector v = sample(length, 2.1);
    Vector projected(length);
    evaluate_projection(cone, v, projected);
    const Vector residual = projected - v;
    CHECK_CLOSE(projected.dot(residual), 0.0, 1e-7);
  }
}

void the_projection_is_nonexpansive() {
  for (const Cone& cone : every_cone()) {
    const Index length = rows(cone);
    const Vector u = sample(length, 0.9);
    const Vector v = sample(length, 1.6);
    Vector image_u(length), image_v(length);
    evaluate_projection(cone, u, image_u);
    evaluate_projection(cone, v, image_v);
    CHECK((image_u - image_v).norm() <= (u - v).norm() + 1e-9);
  }
}

void margin_is_a_shift() {
  for (const Cone& cone : every_cone()) {
    if (!has_interior(cone)) continue;
    const Index length = rows(cone);
    Vector direction(length);
    interior_direction(cone, ConeSide::Primal, direction);
    Vector point = sample(length, 0.4);
    const Scalar before = margin(cone, ConeSide::Primal, point);
    const Scalar shift = 0.75;
    Vector moved = point + shift * direction;
    const Scalar after = margin(cone, ConeSide::Primal, moved);
    CHECK_CLOSE(after, before + shift, 1e-6);
  }
}

void the_zero_cone_has_no_interior() {
  const Cone zero = Zero{3};
  CHECK(!has_interior(zero));
  CHECK(degree(zero) == 0);
  CHECK(!is_curved(zero));
}

void curvature_matches_the_cone_kind() {
  CHECK(!is_curved(Cone{Zero{2}}));
  CHECK(!is_curved(Cone{Nonneg{2}}));
  CHECK(is_curved(Cone{SecondOrder{3}}));
  CHECK(is_curved(Cone{PSDTriangle::of_side(2)}));
  CHECK(is_curved(Cone{Exponential{}}));
}

void a_feasible_step_stays_feasible() {
  for (const Cone& cone : every_cone()) {
    if (!has_interior(cone)) continue;
    const Index length = rows(cone);
    Vector point(length);
    interior_direction(cone, ConeSide::Primal, point);
    const Vector step = sample(length, 3.7);
    const Scalar largest =
        largest_feasible_step(cone, ConeSide::Primal, point, step);
    if (!(largest > 0.0) || !std::isfinite(largest)) continue;
    Vector moved = point + 0.9 * largest * step;
    CHECK(contains(cone, ConeSide::Primal, moved));
  }
}

void validate_rejects_malformed_blocks() {
  bool threw = false;
  try {
    validate(Cones{Power{1.5, 3}});
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);
  threw = false;
  try {
    validate(Cones{SecondOrder{1}});
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);
}

void run_all() {
  RUN(rows_match_the_declared_dimension);
  RUN(projection_is_idempotent);
  RUN(projection_lands_in_the_cone);
  RUN(the_projection_is_orthogonal_to_the_residual);
  RUN(the_projection_is_nonexpansive);
  RUN(margin_is_a_shift);
  RUN(the_zero_cone_has_no_interior);
  RUN(curvature_matches_the_cone_kind);
  RUN(a_feasible_step_stays_feasible);
  RUN(validate_rejects_malformed_blocks);
}

}
}

using proxgqp::run_all;
TEST_MAIN
