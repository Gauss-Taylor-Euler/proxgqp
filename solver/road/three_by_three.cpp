#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <Eigen/Eigenvalues>

#include "road/road.hpp"

namespace proxgqp {
namespace {

constexpr Scalar kPinnedEigenvalue = 1e-14;

enum class Allocation { Diagonal, Dense, Expanded };

struct BlockLayout {
  Allocation allocation = Allocation::Diagonal;
  Index start = 0;
  Index length = 0;
  Index auxiliary_start = -1;
  std::vector<Index> operator_positions;
  std::vector<Index> auxiliary_positions;
  std::vector<Index> auxiliary_diagonal;
};

struct ThreeByThree final : Road {
  explicit ThreeByThree(const RoadSettings& given) : settings(given) {}

  void setup(const SparseMatrix& objective_pattern,
             const SparseMatrix& constraint_pattern,
             const Cones& cones) override {
    back = make_backend(resolve_backend(settings.backend, cones), false);
    back->set_threads(settings.max_threads);
    rows_x = objective_pattern.rows();
    rows_s = constraint_pattern.rows();
    if (constraint_pattern.cols() != rows_x)
      throw std::invalid_argument(
          "the constraint pattern's column count does not match the objective");
    const std::vector<Index> offsets = block_offsets(cones);
    if (offsets.back() != rows_s)
      throw std::invalid_argument(
          "the cone list's total dimension does not match the constraint rows");

    layouts.assign(cones.size(), BlockLayout());
    Index auxiliary = 0;
    for (std::size_t j = 0; j < cones.size(); ++j) {
      BlockLayout& layout = layouts[j];
      layout.start = offsets[j];
      layout.length = offsets[j + 1] - offsets[j];
      const GBlock::Kind widest = widest_operator_kind(cones[j]);
      if (widest == GBlock::Kind::Diagonal) {
        layout.allocation = Allocation::Diagonal;
      } else if (widest == GBlock::Kind::DiagPlusLowRank &&
                 layout.length >= settings.expansion_minimum) {
        layout.allocation = Allocation::Expanded;
        layout.auxiliary_start = auxiliary;
        auxiliary += 2;
      } else {
        layout.allocation = Allocation::Dense;
      }
    }
    rows_auxiliary = auxiliary;

    const Index dimension = rows_x + 2 * rows_s + rows_auxiliary;
    const Index slack_base = rows_x;
    const Index equality_base = rows_x + rows_s;
    const Index auxiliary_base = rows_x + 2 * rows_s;

    detail::PatternBuilder builder;
    builder.dimension = dimension;
    builder.reserve(static_cast<std::size_t>(
        objective_pattern.nonZeros() + constraint_pattern.nonZeros() +
        3 * rows_s + 4 * rows_auxiliary + rows_x));

    for (Index column = 0; column < rows_x; ++column) {
      builder.add(column, column);
      for (SparseMatrix::InnerIterator it(objective_pattern, column); it; ++it)
        builder.add(it.row(), column);
    }
    for (std::size_t j = 0; j < layouts.size(); ++j) {
      const BlockLayout& layout = layouts[j];
      if (layout.allocation == Allocation::Dense) {
        for (Index c = 0; c < layout.length; ++c)
          for (Index r = c; r < layout.length; ++r)
            builder.add(slack_base + layout.start + r,
                        slack_base + layout.start + c);
      } else {
        for (Index r = 0; r < layout.length; ++r)
          builder.add(slack_base + layout.start + r,
                      slack_base + layout.start + r);
      }
      if (layout.allocation != Allocation::Expanded) continue;
      for (Index k = 0; k < 2; ++k) {
        const Index aux = auxiliary_base + layout.auxiliary_start + k;
        for (Index r = 0; r < layout.length; ++r)
          builder.add(aux, slack_base + layout.start + r);
        builder.add(aux, aux);
      }
    }
    for (Index column = 0; column < rows_x; ++column)
      for (SparseMatrix::InnerIterator it(constraint_pattern, column); it; ++it)
        builder.add(equality_base + it.row(), column);
    for (Index row = 0; row < rows_s; ++row) {
      builder.add(equality_base + row, slack_base + row);
      builder.add(equality_base + row, equality_base + row);
    }

    system = builder.compress();

    objective_positions.clear();
    for (Index column = 0; column < rows_x; ++column)
      for (SparseMatrix::InnerIterator it(objective_pattern, column); it; ++it)
        if (it.row() >= column)
          objective_positions.push_back(
              detail::position_of(system, it.row(), column));
    diagonal_positions.resize(rows_x);
    for (Index column = 0; column < rows_x; ++column)
      diagonal_positions[column] = detail::position_of(system, column, column);
    constraint_positions.clear();
    for (Index column = 0; column < rows_x; ++column)
      for (SparseMatrix::InnerIterator it(constraint_pattern, column); it; ++it)
        constraint_positions.push_back(
            detail::position_of(system, equality_base + it.row(), column));
    coupling_positions.resize(rows_s);
    equality_positions.resize(rows_s);
    for (Index row = 0; row < rows_s; ++row) {
      coupling_positions[row] =
          detail::position_of(system, equality_base + row, slack_base + row);
      equality_positions[row] =
          detail::position_of(system, equality_base + row, equality_base + row);
    }
    for (std::size_t j = 0; j < layouts.size(); ++j) {
      BlockLayout& layout = layouts[j];
      layout.operator_positions.clear();
      if (layout.allocation == Allocation::Dense) {
        for (Index c = 0; c < layout.length; ++c)
          for (Index r = c; r < layout.length; ++r)
            layout.operator_positions.push_back(detail::position_of(
                system, slack_base + layout.start + r,
                slack_base + layout.start + c));
      } else {
        for (Index r = 0; r < layout.length; ++r)
          layout.operator_positions.push_back(detail::position_of(
              system, slack_base + layout.start + r,
              slack_base + layout.start + r));
      }
      layout.auxiliary_positions.clear();
      layout.auxiliary_diagonal.clear();
      if (layout.allocation != Allocation::Expanded) continue;
      for (Index k = 0; k < 2; ++k) {
        const Index aux = auxiliary_base + layout.auxiliary_start + k;
        for (Index r = 0; r < layout.length; ++r)
          layout.auxiliary_positions.push_back(detail::position_of(
              system, aux, slack_base + layout.start + r));
        layout.auxiliary_diagonal.push_back(
            detail::position_of(system, aux, aux));
      }
    }

    std::fill(system.valuePtr(), system.valuePtr() + system.nonZeros(), 0.0);
    right_hand_side.setZero(dimension);
    solution.setZero(dimension);
    residual.setZero(dimension);
    correction.setZero(dimension);

    Pattern pattern;
    pattern.dimension = dimension;
    pattern.column_start = system.outerIndexPtr();
    pattern.row_index = system.innerIndexPtr();
    back->symbolic(pattern);
  }

  void values(const SparseMatrix& objective, const SparseMatrix& constraint,
              Scalar rho) override {
    for (Index position : objective_positions)
      system.valuePtr()[position] = 0.0;
    for (Index position : diagonal_positions) system.valuePtr()[position] = 0.0;
    std::size_t index = 0;
    for (Index column = 0; column < rows_x; ++column)
      for (SparseMatrix::InnerIterator it(objective, column); it; ++it)
        if (it.row() >= column)
          system.valuePtr()[objective_positions[index++]] += it.value();
    for (Index column = 0; column < rows_x; ++column)
      system.valuePtr()[diagonal_positions[column]] += rho;
    index = 0;
    for (Index column = 0; column < rows_x; ++column)
      for (SparseMatrix::InnerIterator it(constraint, column); it; ++it)
        system.valuePtr()[constraint_positions[index++]] = it.value();
    for (Index row = 0; row < rows_s; ++row)
      system.valuePtr()[coupling_positions[row]] = 1.0;
  }

  void assemble(const std::vector<GBlock>& blocks,
                const std::vector<BlockScaling>&, Scalar,
                const ConstVectorRef& equality_penalty) override {
    Index auxiliary_positive = 0;
    Index auxiliary_negative = 0;
    for (std::size_t j = 0; j < layouts.size(); ++j) {
      const BlockLayout& layout = layouts[j];
      const GBlock& block = blocks[j];
      if (layout.allocation == Allocation::Dense) {
        write_dense(layout, block);
      } else if (layout.allocation == Allocation::Diagonal) {
        write_diagonal(layout, block);
      } else {
        write_expanded(layout, block, auxiliary_positive, auxiliary_negative);
      }
    }
    for (Index row = 0; row < rows_s; ++row)
      system.valuePtr()[equality_positions[row]] = -equality_penalty(row);
  }

  bool factor() override { return back->numeric(system.valuePtr()); }

  void solve(const Residuals& residuals, Direction& direction) override {
    right_hand_side.head(rows_x) = -residuals.stationary_x;
    right_hand_side.segment(rows_x, rows_s) = -residuals.stationary_slack;
    right_hand_side.segment(rows_x + rows_s, rows_s) = residuals.equality;
    right_hand_side.tail(rows_auxiliary).setZero();
    back->solve(right_hand_side.data(), solution.data());
    detail::refine(system, *back, right_hand_side, settings.refinement_budget,
                   solution, residual, correction);
    direction.dx = solution.head(rows_x);
    direction.ds = solution.segment(rows_x, rows_s);
    direction.dy = solution.segment(rows_x + rows_s, rows_s);
  }

  void set_threads(std::size_t requested) override {
    back->set_threads(requested);
  }
  std::size_t threads() const override { return back->threads(); }
  const char* name() const override { return "three_by_three"; }
  const char* backend_name() const override { return back->name(); }

 private:
  void write_diagonal(const BlockLayout& layout, const GBlock& block) {
    if (block.kind != GBlock::Kind::Diagonal)
      throw std::logic_error(
          "a block allocated as diagonal emitted a wider operator; the widest "
          "kind reported at setup was wrong");
    for (Index r = 0; r < layout.length; ++r)
      system.valuePtr()[layout.operator_positions[r]] = block.diagonal(r);
  }

  void write_dense(const BlockLayout& layout, const GBlock& block) {
    std::size_t index = 0;
    for (Index c = 0; c < layout.length; ++c)
      for (Index r = c; r < layout.length; ++r, ++index)
        system.valuePtr()[layout.operator_positions[index]] =
            entry_of(block, r, c);
  }

  void write_expanded(const BlockLayout& layout, const GBlock& block,
                      Index& positive, Index& negative) {
    Scalar delta = 0.0;
    DenseMatrix columns;
    Vector eigenvalues(2);
    if (block.kind == GBlock::Kind::Diagonal) {
      const Scalar first = block.diagonal(0);
      if ((block.diagonal.array() - first).abs().maxCoeff() >
          1e-12 * std::max(1.0, std::abs(first)))
        throw std::logic_error(
            "an expanded block emitted a diagonal that varies across its rows; "
            "the auxiliary form carries one scalar delta");
      delta = first;
      columns.setZero(layout.length, 2);
      eigenvalues.setZero();
    } else if (block.kind == GBlock::Kind::DiagPlusLowRank) {
      delta = block.delta;
      Eigen::SelfAdjointEigenSolver<DenseMatrix> eigen(block.low_rank_middle);
      eigenvalues = eigen.eigenvalues();
      columns = block.low_rank_columns * eigen.eigenvectors();
    } else {
      throw std::logic_error(
          "an expanded block emitted a dense operator, which the two auxiliary "
          "columns cannot carry");
    }

    for (Index r = 0; r < layout.length; ++r)
      system.valuePtr()[layout.operator_positions[r]] = delta;
    for (Index k = 0; k < 2; ++k) {
      const bool pinned = std::abs(eigenvalues(k)) <= kPinnedEigenvalue;
      const Scalar diagonal = pinned ? -1.0 : -1.0 / eigenvalues(k);
      for (Index r = 0; r < layout.length; ++r)
        system.valuePtr()[layout.auxiliary_positions[k * layout.length + r]] =
            pinned ? 0.0 : columns(r, k);
      system.valuePtr()[layout.auxiliary_diagonal[k]] = diagonal;
      if (diagonal > 0.0) ++positive; else ++negative;
    }
  }

  static Scalar entry_of(const GBlock& block, Index row, Index column) {
    switch (block.kind) {
      case GBlock::Kind::Diagonal:
        return row == column ? block.diagonal(row) : 0.0;
      case GBlock::Kind::DiagPlusLowRank:
        return (row == column ? block.delta : 0.0) +
               block.low_rank_columns.row(row) * block.low_rank_middle *
                   block.low_rank_columns.row(column).transpose();
      default:
        return block.dense(row, column);
    }
  }

  RoadSettings settings;
  std::unique_ptr<Backend> back;
  SparseMatrix system;
  std::vector<BlockLayout> layouts;
  std::vector<Index> objective_positions, diagonal_positions;
  std::vector<Index> constraint_positions, coupling_positions;
  std::vector<Index> equality_positions;
  Vector right_hand_side, solution, residual, correction;
  Index rows_x = 0, rows_s = 0, rows_auxiliary = 0;
};

}

std::unique_ptr<Road> make_three_by_three(const RoadSettings& settings) {
  return std::make_unique<ThreeByThree>(settings);
}

}
