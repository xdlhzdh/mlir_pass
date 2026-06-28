#include "AICompiler/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace aicom {

#define GEN_PASS_DEF_STABLEHLOCONSTANTFOLD
#include "Passes.h.inc"

namespace {

static DenseFPElementsAttr getFpConstantAttr(Value value) {
  if (auto cst = value.getDefiningOp<stablehlo::ConstantOp>())
    return dyn_cast<DenseFPElementsAttr>(cst.getValue());
  if (auto cst = value.getDefiningOp<arith::ConstantOp>())
    return dyn_cast<DenseFPElementsAttr>(cst.getValue());
  return {};
}

static LogicalResult foldBinaryConstants(stablehlo::AddOp op,
                                         PatternRewriter &rewriter) {
  auto lhsAttr = getFpConstantAttr(op.getLhs());
  auto rhsAttr = getFpConstantAttr(op.getRhs());
  if (!lhsAttr || !rhsAttr)
    return failure();

  auto resultType = dyn_cast<RankedTensorType>(op.getType());
  if (!resultType || !resultType.getElementType().isF32())
    return failure();
  if (lhsAttr.getType() != rhsAttr.getType() ||
      lhsAttr.getType() != resultType)
    return failure();

  auto lhsVals = lhsAttr.getValues<APFloat>();
  auto rhsVals = rhsAttr.getValues<APFloat>();
  bool lhsSplat = lhsAttr.isSplat();
  bool rhsSplat = rhsAttr.isSplat();
  int64_t numElems = resultType.getNumElements();

  SmallVector<APFloat, 16> folded;
  folded.reserve(numElems);
  for (int64_t i = 0; i < numElems; ++i) {
    APFloat lhs = lhsSplat ? lhsVals[0] : lhsVals[i];
    APFloat rhs = rhsSplat ? rhsVals[0] : rhsVals[i];
    (void)lhs.add(rhs, APFloat::rmNearestTiesToEven);
    folded.push_back(lhs);
  }

  auto foldedAttr = DenseFPElementsAttr::get(resultType, folded);
  rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(op, resultType,
                                                       foldedAttr);
  return success();
}

static LogicalResult foldBinaryConstants(stablehlo::MulOp op,
                                         PatternRewriter &rewriter) {
  auto lhsAttr = getFpConstantAttr(op.getLhs());
  auto rhsAttr = getFpConstantAttr(op.getRhs());
  if (!lhsAttr || !rhsAttr)
    return failure();

  auto resultType = dyn_cast<RankedTensorType>(op.getType());
  if (!resultType || !resultType.getElementType().isF32())
    return failure();
  if (lhsAttr.getType() != rhsAttr.getType() ||
      lhsAttr.getType() != resultType)
    return failure();

  auto lhsVals = lhsAttr.getValues<APFloat>();
  auto rhsVals = rhsAttr.getValues<APFloat>();
  bool lhsSplat = lhsAttr.isSplat();
  bool rhsSplat = rhsAttr.isSplat();
  int64_t numElems = resultType.getNumElements();

  SmallVector<APFloat, 16> folded;
  folded.reserve(numElems);
  for (int64_t i = 0; i < numElems; ++i) {
    APFloat lhs = lhsSplat ? lhsVals[0] : lhsVals[i];
    APFloat rhs = rhsSplat ? rhsVals[0] : rhsVals[i];
    (void)lhs.multiply(rhs, APFloat::rmNearestTiesToEven);
    folded.push_back(lhs);
  }

  auto foldedAttr = DenseFPElementsAttr::get(resultType, folded);
  rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(op, resultType,
                                                       foldedAttr);
  return success();
}

struct FoldAddConstants : public OpRewritePattern<stablehlo::AddOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(stablehlo::AddOp op,
                                PatternRewriter &rewriter) const override {
    return foldBinaryConstants(op, rewriter);
  }
};

struct FoldMulConstants : public OpRewritePattern<stablehlo::MulOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(stablehlo::MulOp op,
                                PatternRewriter &rewriter) const override {
    return foldBinaryConstants(op, rewriter);
  }
};

} // namespace

struct StablehloConstantFoldPass
    : public impl::StablehloConstantFoldBase<StablehloConstantFoldPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FoldAddConstants, FoldMulConstants>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<Pass> createStablehloConstantFoldPass() {
  return std::make_unique<StablehloConstantFoldPass>();
}

} // namespace aicom
} // namespace mlir
