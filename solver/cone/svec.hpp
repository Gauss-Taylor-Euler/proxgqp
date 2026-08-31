#pragma once

#include <cmath>

#include "cone/kernel.hpp"
#include "cone_types.hpp"
#include "types.hpp"

namespace proxgqp::detail {

inline void matrix_from_svec(const PSDTriangle& cone,
                             const ConstVectorRef& packed,
                             DenseMatrix& matrix) {
  const Index side = static_cast<Index>(cone.side);
  const Scalar inverse_root_two = 1.0 / std::sqrt(2.0);
  matrix.resize(side, side);
  Index position = 0;
  for (Index column = 0; column < side; ++column)
    for (Index row = column; row < side; ++row, ++position)
      matrix(row, column) = matrix(column, row) =
          (row == column) ? packed(position)
                          : packed(position) * inverse_root_two;
}

inline void svec_from_matrix(const PSDTriangle& cone, const DenseMatrix& matrix,
                             VectorRef packed) {
  const Index side = static_cast<Index>(cone.side);
  const Scalar root_two = std::sqrt(2.0);
  Index position = 0;
  for (Index column = 0; column < side; ++column)
    for (Index row = column; row < side; ++row, ++position)
      packed(position) = (row == column) ? matrix(row, column)
                                         : matrix(row, column) * root_two;
}

}
