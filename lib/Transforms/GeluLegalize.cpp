#include "AICompiler/Passes.h"

#include <cmath>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace aicom {

#define GEN_PASS_DEF_GELULEGALIZE
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

static Value geluInputFromHalfMul(stablehlo::MulOp mulOp) {
  Value lhs = unwrapBroadcast(mulOp.getLhs());
  Value rhs = unwrapBroadcast(mulOp.getRhs());
  if (auto lhsHalf = getScalarF32(lhs); lhsHalf && isApprox(*lhsHalf, 0.5f))
    return mulOp.getRhs();
  if (auto rhsHalf = getScalarF32(rhs); rhsHalf && isApprox(*rhsHalf, 0.5f))
    return mulOp.getLhs();
  return {};
}

static bool isErf(Value value) {
  Operation *op = value.getDefiningOp();
  if (!op)
    return false;
  return llvm::isa<chlo::ErfOp>(op) ||
         op->getName().getStringRef() == "chlo.erf";
}

static bool isGeluInner(Value value, Value input) {
  auto addOp = value.getDefiningOp<stablehlo::AddOp>();
  if (!addOp)
    return false;

  Value erfSide;
  if (auto one = getScalarF32(unwrapBroadcast(addOp.getLhs()));
      one && isApprox(*one, 1.0f)) {
    erfSide = addOp.getRhs();
  } else if (auto one = getScalarF32(unwrapBroadcast(addOp.getRhs()));
             one && isApprox(*one, 1.0f)) {
    erfSide = addOp.getLhs();
  } else {
    return false;
  }

  if (!isErf(erfSide))
    return false;

  auto divOp = erfSide.getDefiningOp()->getOperand(0).getDefiningOp<stablehlo::DivOp>();
  if (!divOp)
    return false;

  Value sqrtSide;
  if (divOp.getLhs() == input)
    sqrtSide = divOp.getRhs();
  else if (divOp.getRhs() == input)
    sqrtSide = divOp.getLhs();
  else
    return false;

  auto sqrtOp = unwrapBroadcast(sqrtSide).getDefiningOp<stablehlo::SqrtOp>();
  if (!sqrtOp)
    return false;

  if (auto two = getScalarF32(unwrapBroadcast(sqrtOp.getOperand()));
      !two || !isApprox(*two, 2.0f))
    return false;

  return true;
}

struct MarkGeluMultiplyPattern : public OpRewritePattern<stablehlo::MulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::MulOp mulOp,
                                PatternRewriter &rewriter) const override {
    if (mulOp->hasAttr("aicom.gelu_canonicalized"))
      return failure();

  Value halfMul;
  Value inner;
  if (auto half = mulOp.getLhs().getDefiningOp<stablehlo::MulOp>()) {
    Value input = geluInputFromHalfMul(half);
    if (!input || !isGeluInner(mulOp.getRhs(), input))
      return failure();
    halfMul = mulOp.getLhs();
    inner = mulOp.getRhs();
    (void)halfMul;
    (void)inner;
  } else if (auto half = mulOp.getRhs().getDefiningOp<stablehlo::MulOp>()) {
    Value input = geluInputFromHalfMul(half);
    if (!input || !isGeluInner(mulOp.getLhs(), input))
      return failure();
  } else {
    return failure();
  }

    rewriter.modifyOpInPlace(mulOp, [&] {
      mulOp->setAttr("aicom.gelu_canonicalized", rewriter.getUnitAttr());
    });
    return success();
  }
};

} // namespace

struct GeluLegalizePass : public impl::GeluLegalizeBase<GeluLegalizePass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<MarkGeluMultiplyPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<Pass> createGeluLegalizePass() {
  return std::make_unique<GeluLegalizePass>();
}

} // namespace aicom
} // namespace mlir
