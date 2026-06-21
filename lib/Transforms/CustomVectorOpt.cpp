#include "AICompiler/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_CUSTOMVECTOROPT
#include "Passes.h.inc"

namespace aicom {

namespace {

struct TransferReadToLoad : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp readOp,
                                PatternRewriter &rewriter) const override {
    if (!readOp.getPermutationMap().isMinorIdentity())
      return failure();

    auto memRefType = dyn_cast<MemRefType>(readOp.getBase().getType());
    if (!memRefType || !memRefType.hasStaticShape())
      return failure();

    SmallVector<Value, 4> indices(readOp.getIndices().begin(),
                                  readOp.getIndices().end());
    Value load = vector::LoadOp::create(rewriter, readOp.getLoc(),
                                        readOp.getVectorType(), readOp.getBase(),
                                        indices);
    rewriter.replaceOp(readOp, load);
    return success();
  }
};

struct TransferWriteToStore : public OpRewritePattern<vector::TransferWriteOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferWriteOp writeOp,
                                PatternRewriter &rewriter) const override {
    if (!writeOp.getPermutationMap().isMinorIdentity())
      return failure();

    auto memRefType = dyn_cast<MemRefType>(writeOp.getBase().getType());
    if (!memRefType || !memRefType.hasStaticShape())
      return failure();

    SmallVector<Value, 4> indices(writeOp.getIndices().begin(),
                                  writeOp.getIndices().end());
    rewriter.replaceOpWithNewOp<vector::StoreOp>(
        writeOp, writeOp.getVector(), writeOp.getBase(), indices);
    return success();
  }
};

struct CustomVectorOptPass
    : public impl::CustomVectorOptBase<CustomVectorOptPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<TransferReadToLoad, TransferWriteToStore>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createCustomVectorOptPass() {
  return std::make_unique<CustomVectorOptPass>();
}

} // namespace aicom
} // namespace mlir
