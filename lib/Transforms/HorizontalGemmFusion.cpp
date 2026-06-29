#include "AICompiler/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace aicom {

#define GEN_PASS_DEF_HORIZONTALGEMMFUSION
#include "Passes.h.inc"

namespace {

static stablehlo::DotGeneralOp getDotGeneral(Value value) {
  return value.getDefiningOp<stablehlo::DotGeneralOp>();
}

struct HorizontalGemmFusionPattern
    : public OpRewritePattern<stablehlo::ConcatenateOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::ConcatenateOp concatOp,
                                PatternRewriter &rewriter) const override {
    if (concatOp->hasAttr("aicom.horizontal_gemm_fused"))
      return failure();

    auto operands = concatOp.getOperands();
    if (operands.size() != 2)
      return failure();

    auto dot0 = getDotGeneral(operands[0]);
    auto dot1 = getDotGeneral(operands[1]);
    if (!dot0 || !dot1)
      return failure();

    if (dot0.getLhs() != dot1.getLhs())
      return failure();

    auto outType = dyn_cast<RankedTensorType>(concatOp.getType());
    if (!outType || outType.getRank() < 1)
      return failure();

    int64_t concatDim = concatOp.getDimension();
    if (concatDim < 0)
      concatDim += outType.getRank();
    if (concatDim != outType.getRank() - 1)
      return failure();

    rewriter.modifyOpInPlace(concatOp, [&] {
      concatOp->setAttr("aicom.horizontal_gemm_fused",
                        rewriter.getUnitAttr());
    });
    return success();
  }
};

} // namespace

struct HorizontalGemmFusionPass
    : public impl::HorizontalGemmFusionBase<HorizontalGemmFusionPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<HorizontalGemmFusionPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<Pass> createHorizontalGemmFusionPass() {
  return std::make_unique<HorizontalGemmFusionPass>();
}

} // namespace aicom
} // namespace mlir
