#include "AICompiler/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace aicom {

#define GEN_PASS_DEF_CUSTOMLINALGTOPARALLELLOOPS
#include "Passes.h.inc"

namespace {

static bool isAllParallel(linalg::GenericOp op) {
  for (auto it : op.getIteratorTypesArray())
    if (it != utils::IteratorType::parallel)
      return false;
  return true;
}

static bool allIdentityMaps(linalg::GenericOp op) {
  for (auto map : op.getIndexingMapsArray())
    if (!map.isIdentity())
      return false;
  return true;
}

static Value constantIndex(OpBuilder &b, Location loc, int64_t v) {
  return arith::ConstantIndexOp::create(b, loc, v);
}

/// Lower elementwise `linalg.generic` (all parallel, identity maps, static
/// memref) to an `scf.parallel` nest. Teaches parallel loop mapping vs
/// `convert-linalg-to-loops` sequential lowering.
struct ElementwiseGenericToParallelLoops
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    if (!isAllParallel(op) || !allIdentityMaps(op))
      return failure();
    if (op.getNumDpsInits() != 1)
      return failure();

    auto outType =
        dyn_cast<MemRefType>(op.getDpsInitOperand(0)->get().getType());
    if (!outType || !outType.hasStaticShape())
      return failure();

    for (Value in : op.getDpsInputs()) {
      auto inTy = dyn_cast<MemRefType>(in.getType());
      if (!inTy || !inTy.hasStaticShape())
        return failure();
    }

    int64_t rank = outType.getRank();
    Location loc = op.getLoc();
    SmallVector<Value> lbs, ubs, steps;
    for (int64_t d = 0; d < rank; ++d) {
      lbs.push_back(constantIndex(rewriter, loc, 0));
      ubs.push_back(constantIndex(rewriter, loc, outType.getDimSize(d)));
      steps.push_back(constantIndex(rewriter, loc, 1));
    }

    Value outMemref = op.getDpsInitOperand(0)->get();
    SmallVector<Value> inMemrefs;
    for (auto &operand : op.getDpsInputOperands())
      inMemrefs.push_back(operand->get());

    scf::ParallelOp parallel = scf::ParallelOp::create(
        rewriter, loc, lbs, ubs, steps,
        [&](OpBuilder &b, Location bodyLoc, ValueRange ivs) {
          IRMapping map;
          Block &oldBlock = op.getRegion().front();

          // Bufferized elementwise generics use scalar block args (loaded by
          // linalg); map them to memref.load at the parallel indices.
          unsigned argIdx = 0;
          for (Value inBuf : inMemrefs) {
            Type elemTy = cast<MemRefType>(inBuf.getType()).getElementType();
            Value loaded =
                memref::LoadOp::create(b, bodyLoc, elemTy, inBuf, ivs);
            map.map(oldBlock.getArgument(argIdx++), loaded);
          }
          if (argIdx < oldBlock.getNumArguments()) {
            Type elemTy = outType.getElementType();
            Value loaded =
                memref::LoadOp::create(b, bodyLoc, elemTy, outMemref, ivs);
            map.map(oldBlock.getArgument(argIdx), loaded);
          }

          for (auto &bodyOp : oldBlock.without_terminator()) {
            if (isa<linalg::IndexOp>(bodyOp))
              continue;
            b.clone(bodyOp, map);
          }

          auto yield = cast<linalg::YieldOp>(oldBlock.getTerminator());
          Value result = map.lookupOrDefault(yield.getOperand(0));
          memref::StoreOp::create(b, bodyLoc, result, outMemref, ivs);
        });

    rewriter.replaceOp(op, parallel.getResults());
    return success();
  }
};

/// Lower `linalg.matmul` to parallel loops over (i, j) + reduction loop on k.
struct MatmulToParallelLoops : public OpRewritePattern<linalg::MatmulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op.getDpsInputOperand(0)->get();
    Value rhs = op.getDpsInputOperand(1)->get();
    Value out = op.getDpsInitOperand(0)->get();

    auto lhsTy = dyn_cast<MemRefType>(lhs.getType());
    auto rhsTy = dyn_cast<MemRefType>(rhs.getType());
    auto outTy = dyn_cast<MemRefType>(out.getType());
    if (!lhsTy || !rhsTy || !outTy)
      return failure();
    if (!lhsTy.hasStaticShape() || !rhsTy.hasStaticShape() ||
        !outTy.hasStaticShape())
      return failure();
    if (lhsTy.getRank() != 2 || rhsTy.getRank() != 2 || outTy.getRank() != 2)
      return failure();

    int64_t m = lhsTy.getDimSize(0);
    int64_t kDim = lhsTy.getDimSize(1);
    int64_t n = rhsTy.getDimSize(1);
    if (rhsTy.getDimSize(0) != kDim)
      return failure();

    Location loc = op.getLoc();
    Value c0 = constantIndex(rewriter, loc, 0);
    Value c1 = constantIndex(rewriter, loc, 1);
    Value mUb = constantIndex(rewriter, loc, m);
    Value nUb = constantIndex(rewriter, loc, n);
    Value kUb = constantIndex(rewriter, loc, kDim);

    Type elemTy = outTy.getElementType();
    Value zeroVal = mlir::isa<FloatType>(elemTy)
                        ? arith::ConstantOp::create(
                              rewriter, loc, elemTy,
                              rewriter.getFloatAttr(elemTy, 0.0))
                        : arith::ConstantOp::create(rewriter, loc, elemTy,
                                                     rewriter.getZeroAttr(elemTy));

    scf::ParallelOp parallel = scf::ParallelOp::create(
        rewriter, loc, ValueRange{c0, c0}, ValueRange{mUb, nUb},
        ValueRange{c1, c1},
        [&](OpBuilder &b, Location bodyLoc, ValueRange ivs) {
          Value i = ivs[0];
          Value j = ivs[1];
          auto loop = scf::ForOp::create(b, bodyLoc, c0, kUb, c1,
                                         ValueRange{zeroVal});
          b.setInsertionPointToStart(loop.getBody());
          Value k = loop.getInductionVar();
          Value acc = loop.getRegionIterArgs()[0];
          Value a =
              memref::LoadOp::create(b, bodyLoc, elemTy, lhs, ValueRange{i, k});
          Value bv =
              memref::LoadOp::create(b, bodyLoc, elemTy, rhs, ValueRange{k, j});
          Value prod = arith::MulFOp::create(b, bodyLoc, a, bv);
          Value sum = arith::AddFOp::create(b, bodyLoc, acc, prod);
          scf::YieldOp::create(b, bodyLoc, sum);
          b.setInsertionPointAfter(loop);
          memref::StoreOp::create(b, bodyLoc, loop.getResult(0), out,
                                  ValueRange{i, j});
        });

    rewriter.replaceOp(op, parallel.getResults());
    return success();
  }
};

struct CustomLinalgToParallelLoopsPass
    : public impl::CustomLinalgToParallelLoopsBase<
          CustomLinalgToParallelLoopsPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ElementwiseGenericToParallelLoops, MatmulToParallelLoops>(
        &getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createCustomLinalgToParallelLoopsPass() {
  return std::make_unique<CustomLinalgToParallelLoopsPass>();
}

} // namespace aicom
} // namespace mlir
