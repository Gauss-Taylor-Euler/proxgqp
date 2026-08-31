#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "backend/backend.hpp"
#include "cone/kernel.hpp"
#include "cone_types.hpp"
#include "gblock.hpp"
#include "types.hpp"

namespace proxgqp {

struct Residuals {
  ConstVectorRef stationary_x;
  ConstVectorRef stationary_slack;
  ConstVectorRef equality;
};

struct Direction {
  VectorRef dx;
  VectorRef ds;
  VectorRef dy;
};

enum class RoadKind { ThreeByThree, Schur };

struct RoadSettings {
  BackendKind backend = BackendKind::Auto;
  std::size_t refinement_budget = 3;
  Index expansion_minimum = 8;
  std::size_t max_threads = 1;
};

struct Road {
  virtual ~Road() = default;

  virtual void setup(const SparseMatrix& objective_pattern,
                     const SparseMatrix& constraint_pattern,
                     const Cones& cones) = 0;
  virtual void values(const SparseMatrix& objective,
                      const SparseMatrix& constraint, Scalar rho) = 0;
  virtual void assemble(const std::vector<GBlock>& blocks,
                        const std::vector<BlockScaling>& scalings,
                        Scalar proximal_slack,
                        const ConstVectorRef& equality_penalty) = 0;
  virtual bool factor() = 0;
  virtual void solve(const Residuals& residuals, Direction& direction) = 0;

  virtual void set_threads(std::size_t requested) = 0;
  virtual std::size_t threads() const = 0;
  virtual const char* name() const = 0;
  virtual const char* backend_name() const = 0;
};

std::unique_ptr<Road> make_road(RoadKind kind, const RoadSettings& settings);
std::unique_ptr<Road> make_three_by_three(const RoadSettings& settings);
std::unique_ptr<Road> make_schur(const RoadSettings& settings);

namespace detail {

struct PatternBuilder {
  Index dimension = 0;
  std::vector<Eigen::Triplet<Scalar, StorageIndex>> entries;

  void reserve(std::size_t count);
  void add(Index row, Index column);
  SparseMatrix compress() const;
};

Index position_of(const SparseMatrix& lower, Index row, Index column);

bool still_refining(const Vector& residual, Scalar scale, Scalar& previous);

void refine(const SparseMatrix& lower, const Backend& backend,
            const Vector& right_hand_side, std::size_t budget, Vector& solution,
            Vector& residual, Vector& correction);

}
}
