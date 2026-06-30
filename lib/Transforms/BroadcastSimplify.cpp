#include "AICompiler/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace aicom {

#define GEN_PASS_DEF_BROADCASTSIMPLIFY
#include "Passes.h.inc"

namespace {

static bool sameShape(RankedTensorType lhs, RankedTensorType rhs) {
  if (!lhs || !rhs || lhs.getRank() != rhs.getRank())
    return false;
  return lhs.getShape() == rhs.getShape();
}

static void markSimplified(Operation *op, PatternRewriter &rewriter) {
  if (!op->hasAttr("aicom.broadcast_simplified"))
    rewriter.modifyOpInPlace(op, [&] {
      op->setAttr("aicom.broadcast_simplified", rewriter.getUnitAttr());
    });
}

struct IdentityBroadcastPattern
    : public OpRewritePattern<stablehlo::BroadcastInDimOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::BroadcastInDimOp bcast,
                                PatternRewriter &rewriter) const override {
    auto inType = dyn_cast<RankedTensorType>(bcast.getOperand().getType());
    auto outType = dyn_cast<RankedTensorType>(bcast.getType());
    if (!inType || !outType || !sameShape(inType, outType))
      return failure();

    rewriter.replaceOp(bcast, bcast.getOperand());
    return success();
  }
};

struct NestedBroadcastPattern
    : public OpRewritePattern<stablehlo::BroadcastInDimOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::BroadcastInDimOp outer,
                                PatternRewriter &rewriter) const override {
    auto inner = outer.getOperand().getDefiningOp<stablehlo::BroadcastInDimOp>();
    if (!inner)
      return failure();

    auto innerInType =
        dyn_cast<RankedTensorType>(inner.getOperand().getType());
    auto outerOutType = dyn_cast<RankedTensorType>(outer.getType());
    if (!innerInType || !outerOutType)
      return failure();

    auto innerDims = inner.getBroadcastDimensions();
    auto outerDims = outer.getBroadcastDimensions();
    if (innerDims.size() != static_cast<size_t>(innerInType.getRank()) ||
        outerDims.size() != static_cast<size_t>(inner.getType().getRank()))
      return failure();

    SmallVector<int64_t> composed;
    composed.reserve(innerInType.getRank());
    for (int64_t d = 0; d < innerInType.getRank(); ++d) {
      int64_t midDim = innerDims[d];
      if (midDim < 0 || static_cast<size_t>(midDim) >= outerDims.size())
        return failure();
      composed.push_back(outerDims[midDim]);
    }

    auto fused = stablehlo::BroadcastInDimOp::create(
        rewriter, outer.getLoc(), outerOutType, inner.getOperand(),
        rewriter.getDenseI64ArrayAttr(composed));
    markSimplified(fused, rewriter);
    rewriter.replaceOp(outer, fused);
    return success();
  }
};

} // namespace

struct BroadcastSimplifyPass
    : public impl::BroadcastSimplifyBase<BroadcastSimplifyPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<IdentityBroadcastPattern, NestedBroadcastPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<Pass> createBroadcastSimplifyPass() {
  return std::make_unique<BroadcastSimplifyPass>();
}

} // namespace aicom
} // namespace mlir
