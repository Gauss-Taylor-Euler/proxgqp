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

std::unique_ptr<Road> make_road(RoadKind kind, const RoadSettings& settings) {
  switch (kind) {
    case RoadKind::ThreeByThree: return make_three_by_three(settings);
    default: return make_schur(settings);
  }
}

}
