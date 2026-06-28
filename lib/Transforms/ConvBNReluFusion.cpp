#include "AICompiler/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"

#include <limits>

namespace mlir {
namespace aicom {

#define GEN_PASS_DEF_CONVBNRELUFUSION
#include "Passes.h.inc"

namespace {

static DenseFPElementsAttr getFpConstantAttr(Value value) {
  if (auto cst = value.getDefiningOp<stablehlo::ConstantOp>())
    return dyn_cast<DenseFPElementsAttr>(cst.getValue());
  return {};
}

static bool isAllZeroFpConstant(Value value) {
  auto attr = getFpConstantAttr(value);
  if (!attr)
    return false;
  for (APFloat v : attr.getValues<APFloat>())
    if (!v.isZero())
      return false;
  return true;
}

static Value createPositiveInfinityConstant(Location loc, RankedTensorType type,
                                            PatternRewriter &rewriter) {
  SmallVector<float> values(type.getNumElements(),
                            std::numeric_limits<float>::infinity());
  auto attr = DenseFPElementsAttr::get(type, values);
  return stablehlo::ConstantOp::create(rewriter, loc, type, attr);
}

struct ReluMaxToClampPattern : public OpRewritePattern<stablehlo::MaxOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::MaxOp maxOp,
                                PatternRewriter &rewriter) const override {
    Value zero = maxOp.getRhs();
    Value operand = maxOp.getLhs();
    if (!isAllZeroFpConstant(zero)) {
      zero = maxOp.getLhs();
      operand = maxOp.getRhs();
    }
    if (!isAllZeroFpConstant(zero))
      return failure();

    auto resultType = dyn_cast<RankedTensorType>(maxOp.getType());
    if (!resultType || !resultType.getElementType().isF32())
      return failure();

    Value upper = createPositiveInfinityConstant(maxOp.getLoc(), resultType,
                                                 rewriter);
    Value clamp = stablehlo::ClampOp::create(rewriter, maxOp.getLoc(),
                                             resultType, zero, operand, upper);
    rewriter.replaceOp(maxOp, clamp);
    return success();
  }
};

} // namespace

struct ConvBNReluFusionPass
    : public impl::ConvBNReluFusionBase<ConvBNReluFusionPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ReluMaxToClampPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<Pass> createConvBNReluFusionPass() {
  return std::make_unique<ConvBNReluFusionPass>();
}

} // namespace aicom
} // namespace mlir
