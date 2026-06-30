#include "AICompiler/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace aicom {

#define GEN_PASS_DEF_FLASHATTENTIONTILE
#include "Passes.h.inc"

namespace {

struct FlashAttentionTilePattern
    : public OpRewritePattern<stablehlo::DotGeneralOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::DotGeneralOp dotOp,
                                PatternRewriter &rewriter) const override {
    if (!dotOp->hasAttr("aicom.scaled_dot_product_attention"))
      return failure();
    if (dotOp->hasAttr("aicom.flash_tile"))
      return failure();

    rewriter.modifyOpInPlace(dotOp, [&] {
      dotOp->setAttr("aicom.flash_tile", rewriter.getUnitAttr());
      dotOp->setAttr("aicom.flash_tile_m", rewriter.getI64IntegerAttr(64));
      dotOp->setAttr("aicom.flash_tile_n", rewriter.getI64IntegerAttr(64));
    });
    return success();
  }
};

} // namespace

struct FlashAttentionTilePass
    : public impl::FlashAttentionTileBase<FlashAttentionTilePass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FlashAttentionTilePattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<Pass> createFlashAttentionTilePass() {
  return std::make_unique<FlashAttentionTilePass>();
}

} // namespace aicom
} // namespace mlir
