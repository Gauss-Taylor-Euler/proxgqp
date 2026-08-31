#pragma once

#include <cstddef>
#include <stdexcept>
#include <variant>
#include <vector>

#include "types.hpp"

namespace proxgqp {

struct Zero {
  std::size_t dim = 0;
};

struct Nonneg {
  std::size_t dim = 0;
};

struct SecondOrder {
  std::size_t dim = 0;
};

struct Exponential {
  std::size_t dim = 3;
};

struct Power {
  Scalar alpha = 0.5;
  std::size_t dim = 3;
};

class PSDTriangle {
 public:
  static PSDTriangle of_side(std::size_t triangle_side) {
    return PSDTriangle(triangle_side);
  }

  static PSDTriangle of_dim(std::size_t triangle_dim) {
    std::size_t triangle_side = 0;
    while (triangle_side * (triangle_side + 1) / 2 < triangle_dim)
      ++triangle_side;
    if (triangle_side * (triangle_side + 1) / 2 != triangle_dim)
      throw std::invalid_argument(
          "PSDTriangle dimension is not a triangular number");
    return PSDTriangle(triangle_side);
  }

  std::size_t side = 0;
  std::size_t dim = 0;

 private:
  explicit PSDTriangle(std::size_t triangle_side)
      : side(triangle_side), dim(triangle_side * (triangle_side + 1) / 2) {}
};

using Cone =
    std::variant<Zero, Nonneg, SecondOrder, PSDTriangle, Exponential, Power>;
using Cones = std::vector<Cone>;

inline std::size_t dimension(const Cone& cone) {
  return std::visit([](const auto& block) { return block.dim; }, cone);
}

inline std::size_t total_dimension(const Cones& cones) {
  std::size_t total = 0;
  for (const Cone& cone : cones) total += dimension(cone);
  return total;
}

inline void validate(const Cone& cone) {
  if (dimension(cone) == 0)
    throw std::invalid_argument("a cone block of dimension zero");
  if (const auto* lorentz = std::get_if<SecondOrder>(&cone))
    if (lorentz->dim < 2)
      throw std::invalid_argument(
          "a second-order block of fewer than two rows");
  if (const auto* exponential = std::get_if<Exponential>(&cone))
    if (exponential->dim != 3)
      throw std::invalid_argument(
          "an exponential block that is not of size three");
  if (const auto* power = std::get_if<Power>(&cone)) {
    if (power->dim != 3)
      throw std::invalid_argument("a power block that is not of size three");
    if (!(power->alpha > 0.0 && power->alpha < 1.0))
      throw std::invalid_argument("a power exponent outside (0, 1)");
  }
}

inline void validate(const Cones& cones) {
  for (const Cone& cone : cones) validate(cone);
}

}
