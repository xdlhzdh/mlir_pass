#include "AICompiler/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"

#include <limits>

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

static bool isZeroConstant(Value value) {
  DenseFPElementsAttr attr;
  if (!matchPattern(value, m_Constant(&attr)) || !attr.isSplat())
    return false;
  return attr.getSplatValue<APFloat>().isZero();
}

static bool isZeroBroadCast(Value value) {
  if (auto bcast = value.getDefiningOp<stablehlo::BroadcastInDimOp>())
    return isZeroConstant(bcast.getOperand());
  return isZeroConstant(value);
}

/// Fuse maximum(add(x,c), 0) → clamp(0, add, +inf) and erase maximum.
struct ElementwiseReluFoldPattern : public OpRewritePattern<stablehlo::MaxOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::MaxOp maxOp,
                                PatternRewriter &rewriter) const override {
    Value addVal;
    if (isZeroBroadCast(maxOp.getLhs()) && isElementwiseProducer(maxOp.getRhs())) {
      addVal = maxOp.getRhs();
    } else if (isZeroBroadCast(maxOp.getRhs()) &&
               isElementwiseProducer(maxOp.getLhs())) {
      addVal = maxOp.getLhs();
    } else {
      return failure();
    }

    auto addOp = addVal.getDefiningOp();
    if (!addOp)
      return failure();

    auto resultType = cast<RankedTensorType>(maxOp.getType());
    auto posInf = rewriter.getF32FloatAttr(std::numeric_limits<float>::infinity());
    auto zeroAttr =
        DenseElementsAttr::get(resultType, rewriter.getZeroAttr(rewriter.getF32Type()));
    auto infAttr = DenseElementsAttr::get(resultType, posInf);

    Value lo = rewriter.create<stablehlo::ConstantOp>(maxOp.getLoc(), zeroAttr);
    Value hi = rewriter.create<stablehlo::ConstantOp>(maxOp.getLoc(), infAttr);
    auto clamp = rewriter.create<stablehlo::ClampOp>(
        maxOp.getLoc(), resultType, lo, addVal, hi);
    clamp->setAttr("aicom.elementwise_chain_fused", rewriter.getUnitAttr());
    rewriter.replaceOp(maxOp, clamp.getResult());
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
    patterns.add<ElementwiseReluFoldPattern, ElementwiseChainClampPattern>(
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
