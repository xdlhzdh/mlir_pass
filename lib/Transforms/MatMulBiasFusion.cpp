#include "AICompiler/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace aicom {

#define GEN_PASS_DEF_MATMULBIASFUSION
#include "Passes.h.inc"

namespace {

static Value unwrapBroadcast(Value value) {
  if (auto bcast = value.getDefiningOp<stablehlo::BroadcastInDimOp>())
    return bcast.getOperand();
  return value;
}

static bool isConstantBias(Value value) {
  return isa<stablehlo::ConstantOp>(unwrapBroadcast(value).getDefiningOp());
}

struct MatMulBiasFusionPattern : public OpRewritePattern<stablehlo::AddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::AddOp addOp,
                                PatternRewriter &rewriter) const override {
    if (addOp->hasAttr("aicom.matmul_bias_fused"))
      return failure();

    auto isMatMul = [](Value value) {
      return value.getDefiningOp<stablehlo::DotGeneralOp>() != nullptr;
    };

    Value mmVal;
    Value biasVal;
    if (isMatMul(addOp.getLhs()) && isConstantBias(addOp.getRhs())) {
      mmVal = addOp.getLhs();
      biasVal = addOp.getRhs();
    } else if (isMatMul(addOp.getRhs()) && isConstantBias(addOp.getLhs())) {
      mmVal = addOp.getRhs();
      biasVal = addOp.getLhs();
    } else {
      return failure();
    }

    (void)mmVal;
    (void)biasVal;
    rewriter.modifyOpInPlace(addOp, [&] {
      addOp->setAttr("aicom.matmul_bias_fused", rewriter.getUnitAttr());
    });
    return success();
  }
};

} // namespace

struct MatMulBiasFusionPass : public impl::MatMulBiasFusionBase<MatMulBiasFusionPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<MatMulBiasFusionPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<Pass> createMatMulBiasFusionPass() {
  return std::make_unique<MatMulBiasFusionPass>();
}

} // namespace aicom
} // namespace mlir
