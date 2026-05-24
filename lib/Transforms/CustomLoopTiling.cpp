#include "AICompiler/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mlir {
namespace aicom {

#define GEN_PASS_DEF_CUSTOMLOOPTILING
#include "Passes.h.inc"

namespace {

static std::optional<int64_t> getConstantIndex(Value v) {
  if (auto cst = v.getDefiningOp<arith::ConstantIndexOp>())
    return cst.value();
  return std::nullopt;
}

/// Custom pass: strip-mine outermost scf.for with constant bounds.
struct CustomLoopTilingPass
    : public impl::CustomLoopTilingBase<CustomLoopTilingPass> {
  static constexpr int64_t kTileSize = 4;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    SmallVector<scf::ForOp> targets;

    func.walk([&](scf::ForOp forOp) {
      if (forOp->getParentOfType<scf::ForOp>())
        return;
      // Matmul/reduction loops carry iter_args; strip-mining them breaks semantics.
      if (!forOp.getInitArgs().empty())
        return;

      auto lb = getConstantIndex(forOp.getLowerBound());
      auto ub = getConstantIndex(forOp.getUpperBound());
      auto step = getConstantIndex(forOp.getStep());
      if (!lb || !ub || !step || *step <= 0)
        return;

      int64_t trip = (*ub - *lb + *step - 1) / *step;
      if (trip > kTileSize)
        targets.push_back(forOp);
    });

    for (scf::ForOp forOp : targets)
      tileLoop(forOp);
  }

  void tileLoop(scf::ForOp forOp) {
    OpBuilder builder(forOp);
    Location loc = forOp.getLoc();

    int64_t lb = *getConstantIndex(forOp.getLowerBound());
    int64_t ub = *getConstantIndex(forOp.getUpperBound());
    int64_t step = *getConstantIndex(forOp.getStep());
    int64_t tileStep = kTileSize * step;

    Value outerLB = arith::ConstantIndexOp::create(builder, loc, lb);
    Value outerUB = arith::ConstantIndexOp::create(builder, loc, ub);
    Value outerStep = arith::ConstantIndexOp::create(builder, loc, tileStep);

    auto outerFor =
        scf::ForOp::create(builder, loc, outerLB, outerUB, outerStep);
    builder.setInsertionPointToStart(outerFor.getBody());

    Value ii = outerFor.getInductionVar();
    Value tileSizeVal = arith::ConstantIndexOp::create(builder, loc, tileStep);
    Value innerUB =
        arith::MinSIOp::create(builder, loc,
                               arith::AddIOp::create(builder, loc, ii, tileSizeVal),
                               outerUB);
    Value innerStep = arith::ConstantIndexOp::create(builder, loc, step);

    auto innerFor = scf::ForOp::create(builder, loc, ii, innerUB, innerStep);
    Block *innerBody = innerFor.getBody();
    Block *origBody = forOp.getBody();

    origBody->getArgument(0).replaceAllUsesWith(innerBody->getArgument(0));

    auto &innerOps = innerBody->getOperations();
    auto &origOps = origBody->getOperations();
    innerOps.splice(innerOps.begin(), origOps, origOps.begin(),
                    std::prev(origOps.end()));

    forOp.erase();
  }
};

} // namespace

std::unique_ptr<Pass> createCustomLoopTilingPass() {
  return std::make_unique<CustomLoopTilingPass>();
}

} // namespace aicom
} // namespace mlir
