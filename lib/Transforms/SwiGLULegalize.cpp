#include "AICompiler/Passes.h"

#include <cmath>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace aicom {

#define GEN_PASS_DEF_SWIGLULEGALIZE
#include "Passes.h.inc"

namespace {

static std::optional<float> getScalarF32(Value value) {
  if (auto cst = value.getDefiningOp<stablehlo::ConstantOp>()) {
    if (auto attr = dyn_cast<DenseFPElementsAttr>(cst.getValue())) {
      if (attr.isSplat() && attr.getNumElements() == 1)
        return attr.getSplatValue<APFloat>().convertToFloat();
    }
  }
  return std::nullopt;
}

static bool isApprox(float a, float b, float eps = 1e-6f) {
  return std::fabs(a - b) <= eps;
}

static Value unwrapBroadcast(Value value) {
  if (auto bcast = value.getDefiningOp<stablehlo::BroadcastInDimOp>())
    return bcast.getOperand();
  return value;
}

static bool isOne(Value value) {
  if (auto one = getScalarF32(unwrapBroadcast(value)))
    return isApprox(*one, 1.0f);
  return false;
}

static bool matchSilu(Value siluValue, Value &gateOut) {
  auto siluMul = siluValue.getDefiningOp<stablehlo::MulOp>();
  if (!siluMul)
    return false;

  Value gate;
  Value sigmoid;
  if (siluMul.getLhs().getDefiningOp<stablehlo::DivOp>()) {
    sigmoid = siluMul.getLhs();
    gate = siluMul.getRhs();
  } else if (siluMul.getRhs().getDefiningOp<stablehlo::DivOp>()) {
    sigmoid = siluMul.getRhs();
    gate = siluMul.getLhs();
  } else {
    return false;
  }

  auto sigmoidDiv = sigmoid.getDefiningOp<stablehlo::DivOp>();
  if (!sigmoidDiv || !isOne(sigmoidDiv.getLhs()))
    return false;

  auto denomAdd = sigmoidDiv.getRhs().getDefiningOp<stablehlo::AddOp>();
  if (!denomAdd || !isOne(denomAdd.getLhs()))
    return false;

  auto expOp = denomAdd.getRhs().getDefiningOp<stablehlo::ExpOp>();
  if (!expOp)
    return false;

  auto negOp = expOp.getOperand().getDefiningOp<stablehlo::NegOp>();
  if (!negOp)
    return false;

  if (negOp.getOperand() != gate)
    return false;

  gateOut = gate;
  return true;
}

struct MarkSwiGLUMultiplyPattern : public OpRewritePattern<stablehlo::MulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::MulOp mulOp,
                                PatternRewriter &rewriter) const override {
    if (mulOp->hasAttr("aicom.swiglu_canonicalized"))
      return failure();

    Value gate;
    if (matchSilu(mulOp.getLhs(), gate)) {
      (void)gate;
    } else if (matchSilu(mulOp.getRhs(), gate)) {
      (void)gate;
    } else {
      return failure();
    }

    rewriter.modifyOpInPlace(mulOp, [&] {
      mulOp->setAttr("aicom.swiglu_canonicalized", rewriter.getUnitAttr());
    });
    return success();
  }
};

} // namespace

struct SwiGLULegalizePass
    : public impl::SwiGLULegalizeBase<SwiGLULegalizePass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<MarkSwiGLUMultiplyPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<Pass> createSwiGLULegalizePass() {
  return std::make_unique<SwiGLULegalizePass>();
}

} // namespace aicom
} // namespace mlir
