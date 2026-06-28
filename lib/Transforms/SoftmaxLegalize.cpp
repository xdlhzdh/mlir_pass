#include "AICompiler/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace aicom {

#define GEN_PASS_DEF_SOFTMAXLEGALIZE
#include "Passes.h.inc"

namespace {

static bool isReduceOf(Value value, Value expectedInput) {
  auto reduce = value.getDefiningOp<stablehlo::ReduceOp>();
  if (!reduce || reduce.getInputs().empty())
    return false;
  return reduce.getInputs().front() == expectedInput;
}

struct MarkSoftmaxDividePattern : public OpRewritePattern<stablehlo::DivOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::DivOp divOp,
                                PatternRewriter &rewriter) const override {
    if (divOp->hasAttr("aicom.softmax_canonicalized"))
      return failure();

    Value numerator = divOp.getLhs();
    auto exp = numerator.getDefiningOp<stablehlo::ExpOp>();
    if (!exp)
      return failure();

    auto denomBroadcast = divOp.getRhs().getDefiningOp<stablehlo::BroadcastInDimOp>();
    if (!denomBroadcast)
      return failure();
    if (!isReduceOf(denomBroadcast.getOperand(), numerator))
      return failure();

    rewriter.modifyOpInPlace(divOp, [&] {
      divOp->setAttr("aicom.softmax_canonicalized", rewriter.getUnitAttr());
    });
    return success();
  }
};

} // namespace

struct SoftmaxLegalizePass
    : public impl::SoftmaxLegalizeBase<SoftmaxLegalizePass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<MarkSoftmaxDividePattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<Pass> createSoftmaxLegalizePass() {
  return std::make_unique<SoftmaxLegalizePass>();
}

} // namespace aicom
} // namespace mlir
