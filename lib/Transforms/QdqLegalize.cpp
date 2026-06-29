#include "AICompiler/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace aicom {

#define GEN_PASS_DEF_QDQLEGALIZE
#include "Passes.h.inc"

namespace {

static Value unwrapBroadcast(Value value) {
  if (auto bcast = value.getDefiningOp<stablehlo::BroadcastInDimOp>())
    return bcast.getOperand();
  return value;
}

static bool isDequantizeChain(Value value) {
  auto mulOp = value.getDefiningOp<stablehlo::MulOp>();
  if (!mulOp)
    return false;

  auto subOp = mulOp.getLhs().getDefiningOp<stablehlo::SubtractOp>();
  if (!subOp)
    subOp = mulOp.getRhs().getDefiningOp<stablehlo::SubtractOp>();
  if (!subOp)
    return false;

  Value subLhs = subOp.getLhs();
  if (auto convertOp = subLhs.getDefiningOp<stablehlo::ConvertOp>())
    subLhs = convertOp.getOperand();

  // Teaching pattern: dq(x) = (x - zp) * scale
  (void)subLhs;
  (void)unwrapBroadcast(subOp.getRhs());
  (void)unwrapBroadcast(mulOp.getLhs() == subOp.getResult() ? mulOp.getRhs()
                                                            : mulOp.getLhs());
  return true;
}

struct MarkQdqMatMulPattern : public OpRewritePattern<stablehlo::DotGeneralOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::DotGeneralOp dotOp,
                                PatternRewriter &rewriter) const override {
    if (dotOp->hasAttr("aicom.qdq_matmul_canonicalized"))
      return failure();

    if (!isDequantizeChain(dotOp.getLhs()) ||
        !isDequantizeChain(dotOp.getRhs()))
      return failure();

    rewriter.modifyOpInPlace(dotOp, [&] {
      dotOp->setAttr("aicom.qdq_matmul_canonicalized", rewriter.getUnitAttr());
    });
    return success();
  }
};

} // namespace

struct QdqLegalizePass : public impl::QdqLegalizeBase<QdqLegalizePass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<MarkQdqMatMulPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<Pass> createQdqLegalizePass() {
  return std::make_unique<QdqLegalizePass>();
}

} // namespace aicom
} // namespace mlir
