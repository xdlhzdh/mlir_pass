#include "AICompiler/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace aicom {

#define GEN_PASS_DEF_ATTENTIONLEGALIZE
#include "Passes.h.inc"

namespace {

static bool hasPriorDotGeneral(stablehlo::DotGeneralOp dot) {
  auto func = dot->getParentOfType<func::FuncOp>();
  if (!func)
    return false;
  bool foundSelf = false;
  for (auto other : func.getOps<stablehlo::DotGeneralOp>()) {
    if (other.getOperation() == dot.getOperation()) {
      foundSelf = true;
      continue;
    }
    if (!foundSelf)
      return true;
  }
  return false;
}

struct MarkScaledDotProductAttentionPattern
    : public OpRewritePattern<stablehlo::DotGeneralOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::DotGeneralOp dotOp,
                                PatternRewriter &rewriter) const override {
    if (dotOp->hasAttr("aicom.scaled_dot_product_attention"))
      return failure();
    if (!hasPriorDotGeneral(dotOp))
      return failure();

    auto softmaxDiv = dotOp.getLhs().getDefiningOp<stablehlo::DivOp>();
    if (!softmaxDiv ||
        !softmaxDiv->hasAttr("aicom.softmax_canonicalized"))
      return failure();

    rewriter.modifyOpInPlace(dotOp, [&] {
      dotOp->setAttr("aicom.scaled_dot_product_attention",
                     rewriter.getUnitAttr());
    });
    return success();
  }
};

} // namespace

struct AttentionLegalizePass
    : public impl::AttentionLegalizeBase<AttentionLegalizePass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<MarkScaledDotProductAttentionPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<Pass> createAttentionLegalizePass() {
  return std::make_unique<AttentionLegalizePass>();
}

} // namespace aicom
} // namespace mlir
