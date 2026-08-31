#include "gblock.hpp"

#include <cassert>
#include <cmath>
#include <stdexcept>

#include <Eigen/Cholesky>
#include <Eigen/LU>

namespace proxgqp {
namespace {

constexpr Scalar kUniformTolerance = 1e-12;

bool is_uniform(const ConstVectorRef& values) {
  if (values.size() <= 1) return true;
  const Scalar first = values(0);
  const Scalar scale = std::max(1.0, std::abs(first));
  return (values.array() - first).abs().maxCoeff() <= kUniformTolerance * scale;
}

}

Index block_rows(const GBlock& block) {
  switch (block.kind) {
    case GBlock::Kind::Diagonal: return block.diagonal.size();
    case GBlock::Kind::DiagPlusLowRank: return block.low_rank_columns.rows();
    default: return block.dense.rows();
  }
}

void apply_block(const GBlock& block, const ConstVectorRef& v, VectorRef out) {
  switch (block.kind) {
    case GBlock::Kind::Diagonal:
      out = block.diagonal.cwiseProduct(v);
      break;
    case GBlock::Kind::DiagPlusLowRank:
      out.noalias() = block.low_rank_columns *
                      (block.low_rank_middle *
                       (block.low_rank_columns.transpose() * v));
      out += block.delta * v;
      break;
    default:
      out.noalias() = block.dense * v;
      break;
  }
}

void apply_all(const std::vector<GBlock>& blocks,
               const std::vector<Index>& offsets, const ConstVectorRef& v,
               Vector& out) {
  assert(v.data() != out.data());
  for (std::size_t index = 0; index < blocks.size(); ++index) {
    const Index start = offsets[index];
    const Index length = offsets[index + 1] - offsets[index];
    apply_block(blocks[index], v.segment(start, length),
                out.segment(start, length));
  }
}

void invert_block(const GBlock& block, GBlock& out) {
  out.kind = block.kind;
  switch (block.kind) {
    case GBlock::Kind::Diagonal:
      out.diagonal = block.diagonal.cwiseInverse();
      break;
    case GBlock::Kind::DiagPlusLowRank: {
      const DenseMatrix& columns = block.low_rank_columns;
      const DenseMatrix& middle = block.low_rank_middle;
      const Index rank = middle.rows();
      const DenseMatrix gram = columns.transpose() * columns;
      const DenseMatrix inner =
          block.delta * DenseMatrix::Identity(rank, rank) + middle * gram;
      DenseMatrix corrected = inner.fullPivLu().solve(middle);
      corrected = 0.5 * (corrected + corrected.transpose()).eval();
      out.delta = 1.0 / block.delta;
      out.low_rank_columns = columns;
      out.low_rank_middle = -corrected / block.delta;
      break;
    }
    default: {
      const Index rows = block.dense.rows();
      const DenseMatrix identity = DenseMatrix::Identity(rows, rows);
      Eigen::LLT<DenseMatrix> cholesky(block.dense);
      if (cholesky.info() == Eigen::Success)
        out.dense = cholesky.solve(identity);
      else
        out.dense = block.dense.fullPivLu().solve(identity);
      out.dense = 0.5 * (out.dense + out.dense.transpose()).eval();
      break;
    }
  }
}

void add_penalty(const ConstVectorRef& penalty, GBlock& block) {
  switch (block.kind) {
    case GBlock::Kind::Diagonal:
      block.diagonal += penalty;
      break;
    case GBlock::Kind::DiagPlusLowRank:
      if (!is_uniform(penalty))
        throw std::invalid_argument(
            "a diagonal-plus-low-rank block was handed a penalty that varies "
            "across its rows; the scalar delta cannot absorb it. On a curved "
            "cone the penalty is block-constant, which is what makes this "
            "form closed under the Schur weight");
      block.delta += penalty(0);
      break;
    default:
      block.dense.diagonal() += penalty;
      break;
  }
}

void schur_weight(const GBlock& block, const ConstVectorRef& equality_penalty,
                  GBlock& out) {
  GBlock inverted;
  invert_block(block, inverted);
  add_penalty(equality_penalty, inverted);
  invert_block(inverted, out);
}

}
