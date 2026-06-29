#include "AICompiler/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace aicom {

#define GEN_PASS_DEF_ELEMENTWISECHAINLEGALIZE
#include "Passes.h.inc"

namespace {

static bool isElementwiseProducer(Value value) {
  Operation *op = value.getDefiningOp();
  if (!op)
    return false;
  return isa<stablehlo::AddOp, stablehlo::MulOp>(op);
}

struct ElementwiseChainPattern : public OpRewritePattern<stablehlo::MaxOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::MaxOp maxOp,
                                PatternRewriter &rewriter) const override {
    if (maxOp->hasAttr("aicom.elementwise_chain_fused"))
      return failure();

    if (!isElementwiseProducer(maxOp.getLhs()) &&
        !isElementwiseProducer(maxOp.getRhs()))
      return failure();

    rewriter.modifyOpInPlace(maxOp, [&] {
      maxOp->setAttr("aicom.elementwise_chain_fused", rewriter.getUnitAttr());
    });
    return success();
  }
};

struct ElementwiseChainClampPattern
    : public OpRewritePattern<stablehlo::ClampOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::ClampOp clampOp,
                                PatternRewriter &rewriter) const override {
    if (clampOp->hasAttr("aicom.elementwise_chain_fused"))
      return failure();

    if (!isElementwiseProducer(clampOp.getOperand()))
      return failure();

    rewriter.modifyOpInPlace(clampOp, [&] {
      clampOp->setAttr("aicom.elementwise_chain_fused", rewriter.getUnitAttr());
    });
    return success();
  }
};

} // namespace

struct ElementwiseChainLegalizePass
    : public impl::ElementwiseChainLegalizeBase<ElementwiseChainLegalizePass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ElementwiseChainPattern, ElementwiseChainClampPattern>(
        &getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<Pass> createElementwiseChainLegalizePass() {
  return std::make_unique<ElementwiseChainLegalizePass>();
}

} // namespace aicom
} // namespace mlir
