#include "AICompiler/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace aicom {

#define GEN_PASS_DEF_PRODUCERCONSUMERLEGALIZE
#include "Passes.h.inc"

namespace {

static stablehlo::DotGeneralOp findDotGeneralProducer(Value value) {
  if (auto dot = value.getDefiningOp<stablehlo::DotGeneralOp>())
    return dot;
  if (auto mul = value.getDefiningOp<stablehlo::MulOp>()) {
    if (auto dot = mul.getLhs().getDefiningOp<stablehlo::DotGeneralOp>())
      return dot;
    if (auto dot = mul.getRhs().getDefiningOp<stablehlo::DotGeneralOp>())
      return dot;
  }
  return {};
}

/// Fuse dot → subtract(max_broadcast) → exp by feeding exp from dot and max
/// directly (erase subtract when it only exists for numerically-stable exp).
struct ProducerConsumerExpPattern
    : public OpRewritePattern<stablehlo::ExpOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::ExpOp expOp,
                                PatternRewriter &rewriter) const override {
    if (expOp->hasAttr("aicom.producer_consumer_fused"))
      return failure();

    auto subOp = expOp.getOperand().getDefiningOp<stablehlo::SubtractOp>();
    if (!subOp)
      return failure();

    Value scores;
    Value maxBroadcast;
    if (findDotGeneralProducer(subOp.getLhs())) {
      scores = subOp.getLhs();
      maxBroadcast = subOp.getRhs();
    } else if (findDotGeneralProducer(subOp.getRhs())) {
      scores = subOp.getRhs();
      maxBroadcast = subOp.getLhs();
    } else {
      return failure();
    }

    if (!subOp->hasOneUse())
      return failure();

    // exp(scores - max) → exp(scores) * exp(-max)  (teaching graph fold).
    Value negMax =
        stablehlo::NegOp::create(rewriter, expOp.getLoc(), maxBroadcast);
    Value expNeg = stablehlo::ExpOp::create(rewriter, expOp.getLoc(), negMax);
    Value expScores =
        stablehlo::ExpOp::create(rewriter, expOp.getLoc(), scores);
    auto mul = stablehlo::MulOp::create(rewriter, expOp.getLoc(), expScores,
                                        expNeg);
    mul->setAttr("aicom.producer_consumer_fused", rewriter.getUnitAttr());
    rewriter.replaceOp(expOp, mul.getResult());
    if (subOp->use_empty())
      rewriter.eraseOp(subOp);
    return success();
  }
};

} // namespace

struct ProducerConsumerLegalizePass
    : public impl::ProducerConsumerLegalizeBase<ProducerConsumerLegalizePass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ProducerConsumerExpPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<Pass> createProducerConsumerLegalizePass() {
  return std::make_unique<ProducerConsumerLegalizePass>();
}

} // namespace aicom
} // namespace mlir
