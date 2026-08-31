#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "road/road.hpp"

namespace proxgqp {
namespace {

using RowMajorSparse =
    Eigen::SparseMatrix<Scalar, Eigen::RowMajor, StorageIndex>;

struct BlockLayout {
  Index start = 0;
  Index length = 0;
  bool diagonal_weight = false;
  bool by_application = false;
  std::vector<Index> support;
  DenseMatrix rows_of_constraint;
  std::vector<Index> pair_positions;
  std::vector<std::vector<Index>> row_columns;
  std::vector<std::vector<Index>> row_pair_positions;
};

struct Schur final : Road {
  explicit Schur(const RoadSettings& given) : settings(given) {}

  void setup(const SparseMatrix& objective_pattern,
             const SparseMatrix& constraint_pattern,
             const Cones& given_cones) override {
    cones = &given_cones;
    back = make_backend(resolve_backend(settings.backend, given_cones), true);
    back->set_threads(settings.max_threads);
    rows_x = objective_pattern.rows();
    rows_s = constraint_pattern.rows();
    const std::vector<Index> offsets = block_offsets(given_cones);
    if (offsets.back() != rows_s)
      throw std::invalid_argument(
          "the cone list's total dimension does not match the constraint rows");

    const RowMajorSparse by_row(constraint_pattern);
    layouts.assign(given_cones.size(), BlockLayout());
    detail::PatternBuilder builder;
    builder.dimension = rows_x;
    for (Index column = 0; column < rows_x; ++column) {
      builder.add(column, column);
      for (SparseMatrix::InnerIterator it(objective_pattern, column); it; ++it)
        builder.add(it.row(), column);
    }

    for (std::size_t j = 0; j < given_cones.size(); ++j) {
      BlockLayout& layout = layouts[j];
      layout.start = offsets[j];
      layout.length = offsets[j + 1] - offsets[j];
      layout.diagonal_weight =
          widest_operator_kind(given_cones[j]) == GBlock::Kind::Diagonal;
      layout.by_application =
          schur_assembly(given_cones[j]) == SchurAssembly::ByApplication;

      if (layout.diagonal_weight) {
        layout.row_columns.assign(layout.length, {});
        for (Index r = 0; r < layout.length; ++r) {
          for (RowMajorSparse::InnerIterator it(by_row, layout.start + r); it;
               ++it)
            layout.row_columns[r].push_back(it.col());
          for (std::size_t a = 0; a < layout.row_columns[r].size(); ++a)
            for (std::size_t b = 0; b <= a; ++b)
              builder.add(layout.row_columns[r][a], layout.row_columns[r][b]);
        }
        continue;
      }
      std::vector<Index> support;
      for (Index r = 0; r < layout.length; ++r)
        for (RowMajorSparse::InnerIterator it(by_row, layout.start + r); it;
             ++it)
          support.push_back(it.col());
      std::sort(support.begin(), support.end());
      support.erase(std::unique(support.begin(), support.end()), support.end());
      layout.support = support;
      for (std::size_t a = 0; a < support.size(); ++a)
        for (std::size_t b = 0; b <= a; ++b)
          builder.add(support[a], support[b]);
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

    for (BlockLayout& layout : layouts) {
      if (layout.diagonal_weight) {
        layout.row_pair_positions.assign(layout.length, {});
        for (Index r = 0; r < layout.length; ++r)
          for (std::size_t a = 0; a < layout.row_columns[r].size(); ++a)
            for (std::size_t b = 0; b <= a; ++b)
              layout.row_pair_positions[r].push_back(detail::position_of(
                  system, layout.row_columns[r][a], layout.row_columns[r][b]));
        continue;
      }
      layout.pair_positions.clear();
      for (std::size_t a = 0; a < layout.support.size(); ++a)
        for (std::size_t b = 0; b <= a; ++b)
          layout.pair_positions.push_back(
              detail::position_of(system, layout.support[a],
                                  layout.support[b]));
    }

    base_values.assign(static_cast<std::size_t>(system.nonZeros()), 0.0);
    std::copy(base_values.begin(), base_values.end(), system.valuePtr());
    right_hand_side.setZero(rows_x);
    solution.setZero(rows_x);
    residual.setZero(rows_x);
    correction.setZero(rows_x);
    slack_work.setZero(rows_s);
    slack_other.setZero(rows_s);

    Pattern pattern;
    pattern.dimension = rows_x;
    pattern.column_start = system.outerIndexPtr();
    pattern.row_index = system.innerIndexPtr();
    back->symbolic(pattern);
  }

  void values(const SparseMatrix& objective, const SparseMatrix& constraint,
              Scalar rho) override {
    std::fill(base_values.begin(), base_values.end(), 0.0);
    std::size_t index = 0;
    for (Index column = 0; column < rows_x; ++column)
      for (SparseMatrix::InnerIterator it(objective, column); it; ++it)
        if (it.row() >= column)
          base_values[objective_positions[index++]] += it.value();
    for (Index column = 0; column < rows_x; ++column)
      base_values[diagonal_positions[column]] += rho;

    constraint_values = constraint;
    constraint_by_row = RowMajorSparse(constraint);
    for (BlockLayout& layout : layouts) {
      if (layout.diagonal_weight) continue;
      layout.rows_of_constraint.setZero(
          layout.length, static_cast<Index>(layout.support.size()));
      for (Index r = 0; r < layout.length; ++r)
        for (RowMajorSparse::InnerIterator it(constraint_by_row,
                                              layout.start + r);
             it; ++it) {
          const auto found = std::lower_bound(layout.support.begin(),
                                              layout.support.end(), it.col());
          layout.rows_of_constraint(
              r, static_cast<Index>(found - layout.support.begin())) =
              it.value();
        }
    }
  }

  void assemble(const std::vector<GBlock>& blocks,
                const std::vector<BlockScaling>& scalings,
                Scalar proximal_slack,
                const ConstVectorRef& equality_penalty) override {
    slack = proximal_slack;
    block_scalings = &scalings;
    stored_penalty = equality_penalty;
    std::copy(base_values.begin(), base_values.end(), system.valuePtr());
    weights.assign(layouts.size(), GBlock());
    inverses.assign(layouts.size(), GBlock());

    for (std::size_t j = 0; j < layouts.size(); ++j) {
      BlockLayout& layout = layouts[j];
      const auto penalty =
          equality_penalty.segment(layout.start, layout.length);
      if (!layout.by_application) {
        invert_block(blocks[j], inverses[j]);
        schur_weight(blocks[j], penalty, weights[j]);
      }
      if (layout.diagonal_weight) {
        for (Index r = 0; r < layout.length; ++r) {
          const Scalar weight = weights[j].diagonal(r);
          const auto& columns = layout.row_columns[r];
          const auto& positions = layout.row_pair_positions[r];
          std::size_t index = 0;
          for (std::size_t a = 0; a < columns.size(); ++a) {
            const Scalar left = constraint_by_row.coeff(layout.start + r,
                                                        columns[a]);
            for (std::size_t b = 0; b <= a; ++b, ++index)
              system.valuePtr()[positions[index]] +=
                  weight * left *
                  constraint_by_row.coeff(layout.start + r, columns[b]);
          }
        }
        continue;
      }
      const DenseMatrix& constraint_rows = layout.rows_of_constraint;
      const Index support = constraint_rows.cols();
      DenseMatrix weighted(layout.length, support);
      Vector column(layout.length);
      Vector image(layout.length);
      for (Index c = 0; c < support; ++c) {
        column = constraint_rows.col(c);
        apply_weight(j, penalty, column, image);
        weighted.col(c) = image;
      }
      const DenseMatrix contribution =
          constraint_rows.transpose() * weighted;
      std::size_t index = 0;
      for (Index a = 0; a < support; ++a)
        for (Index b = 0; b <= a; ++b, ++index)
          system.valuePtr()[layout.pair_positions[index]] += contribution(a, b);
    }
  }

  bool factor() override { return back->numeric(system.valuePtr()); }

  void solve(const Residuals& residuals, Direction& direction) override {
    for (std::size_t j = 0; j < layouts.size(); ++j) {
      const BlockLayout& layout = layouts[j];
      const auto slack_block =
          residuals.stationary_slack.segment(layout.start, layout.length);
      Vector negated = -slack_block;
      Vector image(layout.length);
      apply_inverse(j, negated, image);
      slack_work.segment(layout.start, layout.length) =
          residuals.equality.segment(layout.start, layout.length) - image;
    }
    right_hand_side = -residuals.stationary_x;
    apply_weight_all(slack_work, slack_other);
    right_hand_side.noalias() += constraint_values.transpose() * slack_other;

    back->solve(right_hand_side.data(), solution.data());
    detail::refine(system, *back, right_hand_side, settings.refinement_budget,
                   solution, residual, correction);
    direction.dx = solution;

    slack_other.noalias() = constraint_values * solution;
    slack_other -= slack_work;
    apply_weight_all(slack_other, slack_work);
    direction.dy = slack_work;

    for (std::size_t j = 0; j < layouts.size(); ++j) {
      const BlockLayout& layout = layouts[j];
      Vector combined =
          -residuals.stationary_slack.segment(layout.start, layout.length) -
          direction.dy.segment(layout.start, layout.length);
      Vector image(layout.length);
      apply_inverse(j, combined, image);
      direction.ds.segment(layout.start, layout.length) = image;
    }
  }

  void set_threads(std::size_t requested) override {
    back->set_threads(requested);
  }
  std::size_t threads() const override { return back->threads(); }
  const char* name() const override { return "schur"; }
  const char* backend_name() const override { return back->name(); }

 private:
  void apply_weight(std::size_t j, const ConstVectorRef& penalty,
                    const ConstVectorRef& v, VectorRef out) {
    if (layouts[j].by_application)
      apply_schur_weight((*cones)[j], (*block_scalings)[j], slack, penalty, v,
                         out);
    else
      apply_block(weights[j], v, out);
  }

  void apply_inverse(std::size_t j, const ConstVectorRef& v, VectorRef out) {
    if (layouts[j].by_application)
      apply_inverse_operator((*cones)[j], (*block_scalings)[j], slack, v, out);
    else
      apply_block(inverses[j], v, out);
  }

  void apply_weight_all(const Vector& v, Vector& out) {
    for (std::size_t j = 0; j < layouts.size(); ++j) {
      const BlockLayout& layout = layouts[j];
      const auto penalty =
          stored_penalty.segment(layout.start, layout.length);
      apply_weight(j, penalty, v.segment(layout.start, layout.length),
                   out.segment(layout.start, layout.length));
    }
  }

  RoadSettings settings;
  std::unique_ptr<Backend> back;
  const Cones* cones = nullptr;
  const std::vector<BlockScaling>* block_scalings = nullptr;
  SparseMatrix system, constraint_values;
  RowMajorSparse constraint_by_row;
  std::vector<Scalar> base_values;
  std::vector<BlockLayout> layouts;
  std::vector<GBlock> weights, inverses;
  std::vector<Index> objective_positions, diagonal_positions;
  Vector right_hand_side, solution, residual, correction;
  Vector slack_work, slack_other, stored_penalty;
  Index rows_x = 0, rows_s = 0;
  Scalar slack = 0.0;
};

}

std::unique_ptr<Road> make_schur(const RoadSettings& settings) {
  return std::make_unique<Schur>(settings);
}

}
