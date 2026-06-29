#include "AICompiler/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace aicom {

#define GEN_PASS_DEF_LAYOUTBRIDGELEGALIZE
#include "Passes.h.inc"

namespace {

static bool isNchwToNhwcPerm(ArrayRef<int64_t> perm) {
  static constexpr int64_t kExpected[] = {0, 2, 3, 1};
  if (perm.size() != 4)
    return false;
  for (size_t i = 0; i < 4; ++i)
    if (perm[i] != kExpected[i])
      return false;
  return true;
}

struct LayoutBridgePattern : public OpRewritePattern<stablehlo::TransposeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::TransposeOp transposeOp,
                                PatternRewriter &rewriter) const override {
    if (transposeOp->hasAttr("aicom.layout_folded"))
      return failure();

    auto perm = transposeOp.getPermutation();
    if (!isNchwToNhwcPerm(perm))
      return failure();

    auto convOp = transposeOp.getOperand().getDefiningOp<stablehlo::ConvolutionOp>();
    if (!convOp)
      return failure();

    rewriter.modifyOpInPlace(transposeOp, [&] {
      transposeOp->setAttr("aicom.layout_folded", rewriter.getUnitAttr());
    });
    return success();
  }
};

} // namespace

struct LayoutBridgeLegalizePass
    : public impl::LayoutBridgeLegalizeBase<LayoutBridgeLegalizePass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<LayoutBridgePattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<Pass> createLayoutBridgeLegalizePass() {
  return std::make_unique<LayoutBridgeLegalizePass>();
}

} // namespace aicom
} // namespace mlir
