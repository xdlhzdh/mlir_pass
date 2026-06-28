#include "AICompiler/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"

#include <cmath>

namespace mlir {
namespace aicom {

#define GEN_PASS_DEF_CONVBNFUSION
#include "Passes.h.inc"

namespace {

static DenseFPElementsAttr getFpConstantAttr(Value value) {
  if (auto cst = value.getDefiningOp<arith::ConstantOp>())
    return dyn_cast<DenseFPElementsAttr>(cst.getValue());
  if (auto cst = value.getDefiningOp<stablehlo::ConstantOp>())
    return dyn_cast<DenseFPElementsAttr>(cst.getValue());
  return {};
}

static int64_t channelIndexForLinearIdx(ArrayRef<int64_t> shape,
                                        int64_t featureIndex,
                                        int64_t linearIdx) {
  int64_t inner = 1;
  for (size_t d = featureIndex + 1; d < shape.size(); ++d)
    inner *= shape[d];
  return (linearIdx / inner) % shape[featureIndex];
}

struct ConvBNFusionPattern
    : public OpRewritePattern<stablehlo::BatchNormInferenceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::BatchNormInferenceOp bnOp,
                                PatternRewriter &rewriter) const override {
    Value convOut = bnOp.getOperand();
    auto convOp = convOut.getDefiningOp<stablehlo::ConvolutionOp>();
    if (!convOp)
      return failure();

    auto gammaAttr = getFpConstantAttr(bnOp.getScale());
    auto betaAttr = getFpConstantAttr(bnOp.getOffset());
    auto meanAttr = getFpConstantAttr(bnOp.getMean());
    auto varAttr = getFpConstantAttr(bnOp.getVariance());
    if (!gammaAttr || !betaAttr || !meanAttr || !varAttr)
      return failure();

    float epsilon = bnOp.getEpsilon().convertToFloat();
    SmallVector<float> scaleFactor;
    SmallVector<float> biasFactor;
    auto gammaVals = gammaAttr.getValues<float>();
    auto betaVals = betaAttr.getValues<float>();
    auto meanVals = meanAttr.getValues<float>();
    auto varVals = varAttr.getValues<float>();

    for (size_t i = 0, n = gammaAttr.size(); i < n; ++i) {
      float scale = gammaVals[i] / std::sqrt(varVals[i] + epsilon);
      scaleFactor.push_back(scale);
      biasFactor.push_back(betaVals[i] - meanVals[i] * scale);
    }

    auto weightAttr = getFpConstantAttr(convOp.getRhs());
    if (!weightAttr)
      return failure();

    SmallVector<float> newWeightVals;
    auto weightVals = weightAttr.getValues<float>();
    int numChannels = static_cast<int>(scaleFactor.size());
    auto weightShape = weightAttr.getType().getShape();
    int64_t weightFeatureIndex = bnOp.getFeatureIndex();
    if (weightFeatureIndex >= static_cast<int64_t>(weightShape.size()) ||
        weightShape[weightFeatureIndex] != numChannels)
      return failure();

    for (size_t i = 0, n = weightVals.size(); i < n; ++i) {
      int64_t ch =
          channelIndexForLinearIdx(weightShape, weightFeatureIndex, i);
      newWeightVals.push_back(weightVals[i] * scaleFactor[ch]);
    }

    auto newWeightAttr =
        DenseFPElementsAttr::get(weightAttr.getType(), newWeightVals);
    Value newWeight = stablehlo::ConstantOp::create(
        rewriter, bnOp.getLoc(), newWeightAttr.getType(), newWeightAttr);

    OperationState convState(bnOp.getLoc(),
                             stablehlo::ConvolutionOp::getOperationName());
    stablehlo::ConvolutionOp::build(
        rewriter, convState, convOp.getResult().getType(),
        ValueRange{convOp.getLhs(), newWeight}, convOp->getAttrs());
    Value fusedConv =
        cast<stablehlo::ConvolutionOp>(rewriter.create(convState)).getResult();

    auto biasType = RankedTensorType::get({numChannels}, rewriter.getF32Type());
    auto biasAttr = DenseFPElementsAttr::get(biasType, biasFactor);
    Value biasConst = stablehlo::ConstantOp::create(
        rewriter, bnOp.getLoc(), biasType, biasAttr);

    auto resultType = cast<RankedTensorType>(bnOp.getResult().getType());
    int64_t featureIndex = bnOp.getFeatureIndex();
    SmallVector<int64_t> broadcastDims = {featureIndex};
    Value biasBroadcast = stablehlo::BroadcastInDimOp::create(
        rewriter, bnOp.getLoc(), resultType, biasConst,
        rewriter.getDenseI64ArrayAttr(broadcastDims));

    Value fused = stablehlo::AddOp::create(rewriter, bnOp.getLoc(), resultType,
                                           fusedConv, biasBroadcast);
    rewriter.replaceOp(bnOp, fused);
    if (convOp->use_empty())
      rewriter.eraseOp(convOp);
    return success();
  }
};

} // namespace

struct ConvBNFusionPass : public impl::ConvBNFusionBase<ConvBNFusionPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ConvBNFusionPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<Pass> createConvBNFusionPass() {
  return std::make_unique<ConvBNFusionPass>();
}

} // namespace aicom
} // namespace mlir
