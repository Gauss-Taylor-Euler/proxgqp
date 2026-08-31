#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCore>

namespace proxgqp {

using Scalar = double;
using StorageIndex = int;
using Index = Eigen::Index;

using SparseMatrix = Eigen::SparseMatrix<Scalar, Eigen::ColMajor, StorageIndex>;
using DenseMatrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

using ConstVectorRef = Eigen::Ref<const Vector>;
using VectorRef = Eigen::Ref<Vector>;


inline Scalar infinity_norm(const ConstVectorRef& v) {
  return v.size() ? v.lpNorm<Eigen::Infinity>() : Scalar(0);
}

inline Scalar smallest_entry(const ConstVectorRef& v, Scalar empty) {
  return v.size() ? v.minCoeff() : empty;
}

}
