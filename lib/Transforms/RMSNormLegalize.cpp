#include "AICompiler/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace aicom {

#define GEN_PASS_DEF_RMSNORMLEGALIZE
#include "Passes.h.inc"

namespace {

static bool isSquareOf(Value value, Value &base) {
  auto mul = value.getDefiningOp<stablehlo::MulOp>();
  if (!mul || mul.getLhs() != mul.getRhs())
    return false;
  base = mul.getLhs();
  return true;
}

static bool isReduceOfSquare(Value value, Value &base) {
  Value tensor = value;
  if (auto reshape = value.getDefiningOp<stablehlo::ReshapeOp>())
    tensor = reshape.getOperand();
  auto reduce = tensor.getDefiningOp<stablehlo::ReduceOp>();
  if (!reduce || reduce.getInputs().empty())
    return false;
  return isSquareOf(reduce.getInputs().front(), base);
}

static bool isMeanOfSquare(Value value, Value &base) {
  auto div = value.getDefiningOp<stablehlo::DivOp>();
  if (!div)
    return false;
  return isReduceOfSquare(div.getLhs(), base);
}

static bool isRmsNormSqrtInput(Value value, Value &base) {
  if (auto sqrt = value.getDefiningOp<stablehlo::SqrtOp>())
    value = sqrt.getOperand();
  auto add = value.getDefiningOp<stablehlo::AddOp>();
  if (!add)
    return false;
  return isMeanOfSquare(add.getLhs(), base) || isMeanOfSquare(add.getRhs(), base);
}

static bool isRmsNormDivide(Value value, Value &base) {
  auto div = value.getDefiningOp<stablehlo::DivOp>();
  if (!div)
    return false;
  auto denomBroadcast =
      div.getRhs().getDefiningOp<stablehlo::BroadcastInDimOp>();
  if (!denomBroadcast)
    return false;
  Value sqrtInput = denomBroadcast.getOperand();
  if (!isRmsNormSqrtInput(sqrtInput, base))
    return false;
  return div.getLhs() == base;
}

struct MarkRMSNormMultiplyPattern
    : public OpRewritePattern<stablehlo::MulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::MulOp mulOp,
                                PatternRewriter &rewriter) const override {
    if (mulOp->hasAttr("aicom.rmsnorm_canonicalized"))
      return failure();

    Value base;
    if (isRmsNormDivide(mulOp.getLhs(), base) ||
        isRmsNormDivide(mulOp.getRhs(), base)) {
      rewriter.modifyOpInPlace(mulOp, [&] {
        mulOp->setAttr("aicom.rmsnorm_canonicalized", rewriter.getUnitAttr());
      });
      return success();
    }
    return failure();
  }
};

} // namespace

struct RMSNormLegalizePass
    : public impl::RMSNormLegalizeBase<RMSNormLegalizePass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<MarkRMSNormMultiplyPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<Pass> createRMSNormLegalizePass() {
  return std::make_unique<RMSNormLegalizePass>();
}

} // namespace aicom
} // namespace mlir
