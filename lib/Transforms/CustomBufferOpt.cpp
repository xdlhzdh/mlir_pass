#include "AICompiler/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace aicom {

#define GEN_PASS_DEF_CUSTOMBUFFEROPT
#include "Passes.h.inc"

namespace {

/// Returns true if `buffer` (or any view of it) may outlive the enclosing
/// function: returned, passed to a call, or forwarded through a branch/yield.
static bool mayEscape(Value buffer) {
  SmallVector<Value> worklist{buffer};
  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    for (Operation *user : v.getUsers()) {
      if (isa<func::ReturnOp, CallOpInterface, BranchOpInterface,
              RegionBranchTerminatorOpInterface>(user))
        return true;
      if (auto view = dyn_cast<ViewLikeOpInterface>(user)) {
        if (view.getViewSource() == v)
          llvm::append_range(worklist, user->getResults());
      }
    }
  }
  return false;
}

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

    // A stack buffer that escapes (e.g. returned via expand_shape) dangles
    // after return and is later free()d by the caller.
    if (mayEscape(allocOp.getResult()))
      return failure();

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
