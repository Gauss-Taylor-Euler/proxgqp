#pragma once

#include <memory>
#include <vector>

#include "cone/kernel.hpp"
#include "cone_types.hpp"
#include "gblock.hpp"
#include "iterate.hpp"
#include "road/road.hpp"
#include "types.hpp"

namespace proxgqp {

struct SmoothingResiduals {
  Vector stationary_x;
  Vector stationary_slack;
  Vector cone;
  Vector equality;

  Scalar largest() const;
};

enum class SmoothingKind { Projection, Barrier };

struct SmoothingSettings {
  Scalar centring_fraction = 0.1;
  bool mehrotra = true;
  Scalar smallest_centring = 1e-6;
  Scalar largest_centring = 1.0;
};

struct Smoothing {
  virtual ~Smoothing() = default;

  virtual void setup(const Cones& cones, Index columns, Index rows) = 0;
  virtual void residuals(const Iterate& iterate, const ProblemData& data) = 0;
  virtual const SmoothingResiduals& current() const = 0;
  virtual const std::vector<GBlock>& blocks() const = 0;
  virtual const std::vector<BlockScaling>& scalings() const = 0;
  virtual Residuals road_residuals() const = 0;
  virtual void recover_dual_step(const ConstVectorRef& ds, VectorRef dz) = 0;
  virtual bool corrector(const Iterate& iterate, const ConstVectorRef& ds,
                         const ConstVectorRef& dz, Scalar primal_step,
                         Scalar dual_step) = 0;
  virtual Scalar centring() const = 0;
  virtual Scalar reference(const Iterate& iterate) const = 0;
  virtual bool requires_interior() const = 0;
  virtual bool one_step_per_outer() const = 0;
  virtual const char* name() const = 0;
};

std::unique_ptr<Smoothing> make_smoothing(SmoothingKind kind,
                                          const SmoothingSettings& settings);
std::unique_ptr<Smoothing> make_barrier(const SmoothingSettings& settings);
std::unique_ptr<Smoothing> make_projection(const SmoothingSettings& settings);

}
