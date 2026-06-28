#include "AICompiler/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace aicom {

#define GEN_PASS_DEF_ROPELEGALIZE
#include "Passes.h.inc"

namespace {

static bool isNegOneBroadcast(Value value) {
  auto broadcast = value.getDefiningOp<stablehlo::BroadcastInDimOp>();
  if (!broadcast)
    return false;
  auto constant = broadcast.getOperand().getDefiningOp<stablehlo::ConstantOp>();
  if (!constant)
    return false;
  auto elements = dyn_cast<DenseElementsAttr>(constant.getValue());
  if (!elements || !elements.isSplat())
    return false;
  return elements.getSplatValue<APFloat>().isExactlyValue(-1.0);
}

static bool isRotateHalf(Value value, Value &base) {
  auto concat = value.getDefiningOp<stablehlo::ConcatenateOp>();
  if (!concat || concat.getInputs().size() != 2)
    return false;

  auto matchHalves = [&](Value negatedHalf, Value plainHalf) -> bool {
    auto mul = negatedHalf.getDefiningOp<stablehlo::MulOp>();
    auto plainSlice = plainHalf.getDefiningOp<stablehlo::SliceOp>();
    if (!mul || !plainSlice)
      return false;

    auto negSlice = mul.getLhs().getDefiningOp<stablehlo::SliceOp>();
    Value negScale = mul.getRhs();
    if (!negSlice) {
      negSlice = mul.getRhs().getDefiningOp<stablehlo::SliceOp>();
      negScale = mul.getLhs();
    }
    if (!negSlice || !isNegOneBroadcast(negScale))
      return false;

    if (negSlice.getOperand() != plainSlice.getOperand())
      return false;
    base = negSlice.getOperand();
    return true;
  };

  return matchHalves(concat.getInputs()[0], concat.getInputs()[1]) ||
         matchHalves(concat.getInputs()[1], concat.getInputs()[0]);
}

static bool isMulOf(Value value, Value base) {
  auto mul = value.getDefiningOp<stablehlo::MulOp>();
  return mul && (mul.getLhs() == base || mul.getRhs() == base);
}

struct MarkRoPEAddPattern : public OpRewritePattern<stablehlo::AddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::AddOp addOp,
                                PatternRewriter &rewriter) const override {
    if (addOp->hasAttr("aicom.rope_canonicalized"))
      return failure();

    auto lhsMul = addOp.getLhs().getDefiningOp<stablehlo::MulOp>();
    auto rhsMul = addOp.getRhs().getDefiningOp<stablehlo::MulOp>();
    if (!lhsMul || !rhsMul)
      return failure();

  Value base;
  if (isRotateHalf(lhsMul.getLhs(), base) || isRotateHalf(lhsMul.getRhs(), base)) {
    if (!isMulOf(addOp.getRhs(), base))
      return failure();
  } else if (isRotateHalf(rhsMul.getLhs(), base) ||
             isRotateHalf(rhsMul.getRhs(), base)) {
    if (!isMulOf(addOp.getLhs(), base))
      return failure();
  } else {
    return failure();
  }

    rewriter.modifyOpInPlace(addOp, [&] {
      addOp->setAttr("aicom.rope_canonicalized", rewriter.getUnitAttr());
    });
    return success();
  }
};

struct RoPELegalizePass : public impl::RoPELegalizeBase<RoPELegalizePass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<MarkRoPEAddPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createRoPELegalizePass() {
  return std::make_unique<RoPELegalizePass>();
}

} // namespace aicom
} // namespace mlir
