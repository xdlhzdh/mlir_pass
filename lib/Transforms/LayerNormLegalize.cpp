#include "AICompiler/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace aicom {

#define GEN_PASS_DEF_LAYERNORMLEGALIZE
#include "Passes.h.inc"

namespace {

static bool isReduceOf(Value value, Value operand) {
  Value tensor = value;
  if (auto reshape = value.getDefiningOp<stablehlo::ReshapeOp>())
    tensor = reshape.getOperand();
  auto reduce = tensor.getDefiningOp<stablehlo::ReduceOp>();
  if (!reduce || reduce.getInputs().empty())
    return false;
  return reduce.getInputs().front() == operand;
}

static bool isMeanOf(Value value, Value input) {
  auto div = value.getDefiningOp<stablehlo::DivOp>();
  if (!div)
    return false;
  return isReduceOf(div.getLhs(), input);
}

static bool isCentered(Value value, Value &input) {
  auto sub = value.getDefiningOp<stablehlo::SubtractOp>();
  if (!sub)
    return false;
  input = sub.getLhs();
  Value meanSide = sub.getRhs();
  if (auto broadcast = meanSide.getDefiningOp<stablehlo::BroadcastInDimOp>())
    meanSide = broadcast.getOperand();
  return isMeanOf(meanSide, input) && sub.getLhs() == input;
}

static bool isSquareOf(Value value, Value &base) {
  auto mul = value.getDefiningOp<stablehlo::MulOp>();
  if (!mul || mul.getLhs() != mul.getRhs())
    return false;
  base = mul.getLhs();
  return true;
}

static bool isReduceOfSquare(Value value, Value &centered) {
  Value tensor = value;
  if (auto reshape = value.getDefiningOp<stablehlo::ReshapeOp>())
    tensor = reshape.getOperand();
  auto reduce = tensor.getDefiningOp<stablehlo::ReduceOp>();
  if (!reduce || reduce.getInputs().empty())
    return false;
  return isSquareOf(reduce.getInputs().front(), centered);
}

static bool isMeanOfSquare(Value value, Value &centered) {
  auto div = value.getDefiningOp<stablehlo::DivOp>();
  if (!div)
    return false;
  return isReduceOfSquare(div.getLhs(), centered);
}

static bool isLayerNormSqrtInput(Value value, Value &centered) {
  if (auto sqrt = value.getDefiningOp<stablehlo::SqrtOp>())
    value = sqrt.getOperand();
  auto add = value.getDefiningOp<stablehlo::AddOp>();
  if (!add)
    return false;
  Value input;
  if (!isCentered(centered, input))
    return false;
  return isMeanOfSquare(add.getLhs(), centered) ||
         isMeanOfSquare(add.getRhs(), centered);
}

static bool isLayerNormDivide(Value value, Value &centered) {
  auto div = value.getDefiningOp<stablehlo::DivOp>();
  if (!div)
    return false;
  Value input;
  if (!isCentered(div.getLhs(), input))
    return false;
  centered = div.getLhs();
  auto denomBroadcast =
      div.getRhs().getDefiningOp<stablehlo::BroadcastInDimOp>();
  if (!denomBroadcast)
    return false;
  return isLayerNormSqrtInput(denomBroadcast.getOperand(), centered);
}

static bool isLayerNormScaledMul(Value value, Value &centered) {
  auto mul = value.getDefiningOp<stablehlo::MulOp>();
  if (!mul)
    return false;
  return isLayerNormDivide(mul.getLhs(), centered) ||
         isLayerNormDivide(mul.getRhs(), centered);
}

struct MarkLayerNormAddPattern : public OpRewritePattern<stablehlo::AddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::AddOp addOp,
                                PatternRewriter &rewriter) const override {
    if (addOp->hasAttr("aicom.layernorm_canonicalized"))
      return failure();

    Value centered;
    if (!isLayerNormScaledMul(addOp.getLhs(), centered) &&
        !isLayerNormScaledMul(addOp.getRhs(), centered))
      return failure();

    rewriter.modifyOpInPlace(addOp, [&] {
      addOp->setAttr("aicom.layernorm_canonicalized", rewriter.getUnitAttr());
    });
    return success();
  }
};

struct LayerNormLegalizePass
    : public impl::LayerNormLegalizeBase<LayerNormLegalizePass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<MarkLayerNormAddPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createLayerNormLegalizePass() {
  return std::make_unique<LayerNormLegalizePass>();
}

} // namespace aicom
} // namespace mlir
