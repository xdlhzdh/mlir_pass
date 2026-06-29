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

    if (!findDotGeneralProducer(subOp.getLhs()) &&
        !findDotGeneralProducer(subOp.getRhs()))
      return failure();

    rewriter.modifyOpInPlace(expOp, [&] {
      expOp->setAttr("aicom.producer_consumer_fused", rewriter.getUnitAttr());
    });
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
