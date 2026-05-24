#include "AICompiler/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <algorithm>
#include <cmath>

namespace mlir {
namespace aicom {

#define GEN_PASS_DEF_CUSTOMLINALGOPT
#include "Passes.h.inc"

namespace {

/// Custom pattern (not MLIR upstream): constant-fold elementwise linalg.generic.
struct FoldConstantElementwiseGeneric
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getNumDpsInits() != 1 || op.getNumDpsInputs() != 2)
      return failure();

    for (auto it : op.getIteratorTypesArray())
      if (it != utils::IteratorType::parallel)
        return failure();

    for (auto map : op.getIndexingMapsArray())
      if (!map.isIdentity())
        return failure();

    SmallVector<DenseElementsAttr> inputAttrs;
    for (Value in : op.getDpsInputs()) {
      DenseElementsAttr attr;
      if (!matchPattern(in, m_Constant(&attr)))
        return failure();
      inputAttrs.push_back(attr);
    }

    Block &body = op.getRegion().front();
    if (body.getOperations().size() != 2)
      return failure();

    Operation &compute = body.front();
    auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
    if (!yield || yield.getNumOperands() != 1 ||
        yield.getOperand(0) != compute.getResult(0))
      return failure();

    auto resultType = dyn_cast<RankedTensorType>(op.getResult(0).getType());
    if (!resultType || !resultType.getElementType().isF32())
      return failure();

    auto lhsVals = inputAttrs[0].getValues<APFloat>();
    auto rhsVals = inputAttrs[1].getValues<APFloat>();
    int64_t numElems = resultType.getNumElements();
    bool lhsSplat = inputAttrs[0].isSplat();
    bool rhsSplat = inputAttrs[1].isSplat();

    SmallVector<float> results;
    results.reserve(numElems);
    for (int64_t i = 0; i < numElems; ++i) {
      float lhs = lhsSplat ? lhsVals[0].convertToFloat()
                           : lhsVals[i].convertToFloat();
      float rhs = rhsSplat ? rhsVals[0].convertToFloat()
                           : rhsVals[i].convertToFloat();
      float out;
      if (isa<arith::AddFOp>(compute))
        out = lhs + rhs;
      else if (isa<arith::MulFOp>(compute))
        out = lhs * rhs;
      else if (isa<arith::SubFOp>(compute))
        out = lhs - rhs;
      else if (isa<arith::MaximumFOp>(compute))
        out = std::max(lhs, rhs);
      else
        return failure();
      results.push_back(out);
    }

    auto folded = DenseFPElementsAttr::get(resultType, results);
    Value cst =
        arith::ConstantOp::create(rewriter, op.getLoc(), resultType, folded);
    rewriter.replaceOp(op, cst);
    return success();
  }
};

struct CustomLinalgOptPass
    : public impl::CustomLinalgOptBase<CustomLinalgOptPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FoldConstantElementwiseGeneric>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createCustomLinalgOptPass() {
  return std::make_unique<CustomLinalgOptPass>();
}

} // namespace aicom
} // namespace mlir
