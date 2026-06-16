#include "AICompiler/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
namespace aicom {

#define GEN_PASS_DEF_CUSTOMAFFINEOPT
#include "Passes.h.inc"

namespace {

/// Strip-mine outermost affine.for loops with constant bounds.
struct CustomAffineOptPass
    : public impl::CustomAffineOptBase<CustomAffineOptPass> {
  static constexpr unsigned kTileSize = 4;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    SmallVector<affine::AffineForOp> targets;

    func.walk([&](affine::AffineForOp forOp) {
      if (forOp->getParentOfType<affine::AffineForOp>())
        return;
      if (forOp.getNumIterOperands() > 0)
        return;
      if (!forOp.hasConstantBounds())
        return;

      int64_t lb = forOp.getConstantLowerBound();
      int64_t ub = forOp.getConstantUpperBound();
      int64_t step = forOp.getStepAsInt();
      if (step <= 0)
        return;

      int64_t trip = (ub - lb + step - 1) / step;
      if (trip > static_cast<int64_t>(kTileSize))
        targets.push_back(forOp);
    });

    for (affine::AffineForOp forOp : targets) {
      SmallVector<affine::AffineForOp, 1> band{forOp};
      if (failed(affine::tilePerfectlyNested(band, {kTileSize})))
        signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> createCustomAffineOptPass() {
  return std::make_unique<CustomAffineOptPass>();
}

} // namespace aicom
} // namespace mlir
