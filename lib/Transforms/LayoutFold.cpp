#include "AICompiler/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace aicom {

#define GEN_PASS_DEF_LAYOUTFOLD
#include "Passes.h.inc"

namespace {

static bool isNchwToNhwcPerm(ArrayRef<int64_t> perm) {
  static constexpr int64_t kExpected[] = {0, 2, 3, 1};
  if (perm.size() != 4)
    return false;
  for (size_t i = 0; i < 4; ++i)
    if (perm[i] != kExpected[i])
      return false;
  return true;
}

static bool isNchwBf01(stablehlo::ConvDimensionNumbersAttr dims) {
  return dims.getInputBatchDimension() == 0 &&
         dims.getInputFeatureDimension() == 1 &&
         dims.getInputSpatialDimensions() == ArrayRef<int64_t>{2, 3} &&
         dims.getKernelOutputFeatureDimension() == 0 &&
         dims.getKernelInputFeatureDimension() == 1 &&
         dims.getKernelSpatialDimensions() == ArrayRef<int64_t>{2, 3} &&
         dims.getOutputBatchDimension() == 0 &&
         dims.getOutputFeatureDimension() == 1 &&
         dims.getOutputSpatialDimensions() == ArrayRef<int64_t>{2, 3};
}

static stablehlo::ConvDimensionNumbersAttr
nhwcDimNumbers(PatternRewriter &rewriter) {
  return stablehlo::ConvDimensionNumbersAttr::get(
      rewriter.getContext(),
      /*inputBatchDimension=*/0, /*inputFeatureDimension=*/3,
      /*inputSpatialDimensions=*/{1, 2},
      /*kernelInputFeatureDimension=*/2, /*kernelOutputFeatureDimension=*/3,
      /*kernelSpatialDimensions=*/{0, 1},
      /*outputBatchDimension=*/0, /*outputFeatureDimension=*/3,
      /*outputSpatialDimensions=*/{1, 2});
}

/// P4-style NCHW [b,f,0,1]: fold transpose into NHWC conv + kernel perm.
struct ConvLayoutFoldNhwcPattern
    : public OpRewritePattern<stablehlo::TransposeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::TransposeOp transposeOp,
                                PatternRewriter &rewriter) const override {
    if (!isNchwToNhwcPerm(transposeOp.getPermutation()))
      return failure();

    auto convOp = transposeOp.getOperand().getDefiningOp<stablehlo::ConvolutionOp>();
    if (!convOp || !isNchwBf01(convOp.getDimensionNumbers()))
      return failure();

    auto weightCst = convOp.getRhs().getDefiningOp<stablehlo::ConstantOp>();
    if (!weightCst)
      return failure();

    auto weightType = dyn_cast<RankedTensorType>(weightCst.getType());
    if (!weightType || weightType.getRank() != 4)
      return failure();

    SmallVector<int64_t> kernelPerm = {2, 3, 1, 0};
    auto transposedWeight = stablehlo::TransposeOp::create(
        rewriter, transposeOp.getLoc(), weightCst.getResult(),
        rewriter.getDenseI64ArrayAttr(kernelPerm));

    auto nhwcDims = nhwcDimNumbers(rewriter);
    SmallVector<NamedAttribute> attrs(convOp->getAttrs().begin(),
                                      convOp->getAttrs().end());
    for (auto &attr : attrs) {
      if (attr.getName() == "dimension_numbers")
        attr.setValue(nhwcDims);
    }

    OperationState convState(transposeOp.getLoc(),
                             stablehlo::ConvolutionOp::getOperationName());
    stablehlo::ConvolutionOp::build(
        rewriter, convState, transposeOp.getType(),
        ValueRange{convOp.getLhs(), transposedWeight.getResult()}, attrs);
    auto fusedConv = cast<stablehlo::ConvolutionOp>(rewriter.create(convState));

    rewriter.modifyOpInPlace(fusedConv, [&] {
      fusedConv->setAttr("aicom.layout_folded", rewriter.getUnitAttr());
    });
    rewriter.replaceOp(transposeOp, fusedConv.getResult());
    if (convOp->use_empty())
      rewriter.eraseOp(convOp);
    return success();
  }
};

/// Teaching LIT [b,0,1,f]: same-rank transpose is layout-only — drop it.
struct ConvLayoutFoldSimplePattern
    : public OpRewritePattern<stablehlo::TransposeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::TransposeOp transposeOp,
                                PatternRewriter &rewriter) const override {
    if (!isNchwToNhwcPerm(transposeOp.getPermutation()))
      return failure();

    auto convOp = transposeOp.getOperand().getDefiningOp<stablehlo::ConvolutionOp>();
    if (!convOp || !convOp.getRhs().getDefiningOp<stablehlo::ConstantOp>())
      return failure();

    if (transposeOp.getType() != convOp.getType())
      return failure();

    rewriter.modifyOpInPlace(convOp, [&] {
      convOp->setAttr("aicom.layout_folded", rewriter.getUnitAttr());
    });
    rewriter.replaceOp(transposeOp, convOp.getResult());
    return success();
  }
};

} // namespace

struct LayoutFoldPass : public impl::LayoutFoldBase<LayoutFoldPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ConvLayoutFoldNhwcPattern, ConvLayoutFoldSimplePattern>(
        &getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<Pass> createLayoutFoldPass() {
  return std::make_unique<LayoutFoldPass>();
}

} // namespace aicom
} // namespace mlir
