#include "AICompiler/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace aicom {

#define GEN_PASS_DEF_CUSTOMBUFFEROPT
#include "Passes.h.inc"

namespace {

/// Custom pattern: memref.alloc -> memref.alloca for small static buffers.
struct PromoteSmallAllocToAlloca : public OpRewritePattern<memref::AllocOp> {
  using OpRewritePattern::OpRewritePattern;

  static constexpr int64_t kMaxStackBytes = 4096;

  LogicalResult matchAndRewrite(memref::AllocOp allocOp,
                                PatternRewriter &rewriter) const override {
    MemRefType type = allocOp.getType();
    if (!type.hasStaticShape())
      return failure();

    int64_t numElements = type.getNumElements();
    unsigned bits = type.getElementType().getIntOrFloatBitWidth();
    int64_t totalBytes = numElements * ((bits + 7) / 8);
    if (totalBytes <= 0 || totalBytes > kMaxStackBytes)
      return failure();

    // Do not stack-promote buffers returned to callers (breaks JIT heap free).
    for (Operation *user : allocOp.getResult().getUsers()) {
      if (isa<func::ReturnOp>(user))
        return failure();
    }

    memref::DeallocOp dealloc;
    for (Operation *user : allocOp.getResult().getUsers()) {
      if (auto d = dyn_cast<memref::DeallocOp>(user))
        dealloc = d;
    }

    Value alloca = memref::AllocaOp::create(rewriter, allocOp.getLoc(), type,
                                             allocOp.getAlignmentAttr());
    rewriter.replaceOp(allocOp, alloca);
    if (dealloc)
      rewriter.eraseOp(dealloc);
    return success();
  }
};

struct CustomBufferOptPass
    : public impl::CustomBufferOptBase<CustomBufferOptPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<PromoteSmallAllocToAlloca>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createCustomBufferOptPass() {
  return std::make_unique<CustomBufferOptPass>();
}

} // namespace aicom
} // namespace mlir
