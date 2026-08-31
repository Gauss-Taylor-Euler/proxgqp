#include "road/road.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace proxgqp {
namespace detail {

void PatternBuilder::reserve(std::size_t count) { entries.reserve(count); }

void PatternBuilder::add(Index row, Index column) {
  if (row < column) std::swap(row, column);
  entries.emplace_back(static_cast<StorageIndex>(row),
                       static_cast<StorageIndex>(column), 1.0);
}

SparseMatrix PatternBuilder::compress() const {
  SparseMatrix lower(dimension, dimension);
  lower.setFromTriplets(entries.begin(), entries.end());
  lower.makeCompressed();
  return lower;
}

Index position_of(const SparseMatrix& lower, Index row, Index column) {
  const StorageIndex* rows = lower.innerIndexPtr();
  const StorageIndex* starts = lower.outerIndexPtr();
  const StorageIndex first = starts[column];
  const StorageIndex last = starts[column + 1];
  const StorageIndex* found =
      std::lower_bound(rows + first, rows + last,
                       static_cast<StorageIndex>(row));
  if (found == rows + last || *found != static_cast<StorageIndex>(row))
    throw std::logic_error(
        "an entry was written that the pattern does not contain; setup and "
        "assemble disagree about the shape of the system");
  return static_cast<Index>(found - rows);
}

bool still_refining(const Vector& residual, Scalar scale, Scalar& previous) {
  const Scalar size = residual.lpNorm<Eigen::Infinity>();
  if (size >= previous ||
      size <= std::numeric_limits<Scalar>::epsilon() * scale)
    return false;
  previous = size;
  return true;
}

void refine(const SparseMatrix& lower, const Backend& backend,
            const Vector& right_hand_side, std::size_t budget, Vector& solution,
            Vector& residual, Vector& correction) {
  const Scalar scale =
      std::max(right_hand_side.lpNorm<Eigen::Infinity>(), Scalar(1));
  Scalar previous = std::numeric_limits<Scalar>::infinity();
  for (std::size_t pass = 0; pass < budget; ++pass) {
    residual.noalias() = lower.selfadjointView<Eigen::Lower>() * solution;
    residual = right_hand_side - residual;
    if (!still_refining(residual, scale, previous)) break;
    backend.solve(residual.data(), correction.data());
    solution += correction;
  }
}

}

constexpr Scalar kSmallestReducedDensity = 0.1;

RoadKind resolve_road(RoadKind requested, Index columns, Index rows,
                      Index constraint_nonzeros) {
  if (requested != RoadKind::Auto) return requested;
  if (columns <= 0 || rows <= 0) return RoadKind::ThreeByThree;
  if (rows <= columns) return RoadKind::ThreeByThree;

  const Scalar entries = Scalar(columns) * Scalar(rows);
  const Scalar constraint_density = Scalar(constraint_nonzeros) / entries;
  const Scalar reduced_density =
      std::min(Scalar(1), constraint_density * constraint_density * Scalar(rows));
  if (reduced_density < kSmallestReducedDensity) return RoadKind::ThreeByThree;
  return RoadKind::Schur;
}

std::unique_ptr<Road> make_road(RoadKind kind, const RoadSettings& settings) {
  switch (kind) {
    case RoadKind::Schur: return make_schur(settings);
    default: return make_three_by_three(settings);
  }
}

}
