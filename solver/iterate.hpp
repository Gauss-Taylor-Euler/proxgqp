#pragma once

#include "types.hpp"

namespace proxgqp {

struct ProblemData {
  const SparseMatrix* P = nullptr;
  const Vector* q = nullptr;
  const SparseMatrix* E = nullptr;
  const Vector* b = nullptr;
};

struct Iterate {
  Vector x, s, z, y;
  Vector x_centre, s_centre, z_centre, y_centre;
  Vector cone_penalty;
  Vector equality_penalty;
  Scalar rho = 0.0;
  Scalar proximal_slack = 0.0;
  Scalar regularisation_floor = 0.0;

  void resize(Index columns, Index rows) {
    x.setZero(columns);
    s.setZero(rows);
    z.setZero(rows);
    y.setZero(rows);
    x_centre.setZero(columns);
    s_centre.setZero(rows);
    z_centre.setZero(rows);
    y_centre.setZero(rows);
    cone_penalty.setOnes(rows);
    equality_penalty.setOnes(rows);
  }
};

}
