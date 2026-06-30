#include "AICompiler/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace aicom {

#define GEN_PASS_DEF_CUSTOMLINALGTILE
#include "Passes.h.inc"

namespace {

struct CustomLinalgTilePass
    : public impl::CustomLinalgTileBase<CustomLinalgTilePass> {
  static constexpr int64_t kTileM = 2;
  static constexpr int64_t kTileN = 2;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    IRRewriter rewriter(&getContext());

    SmallVector<linalg::MatmulOp> targets;
    for (linalg::MatmulOp matmulOp : func.getOps<linalg::MatmulOp>()) {
      if (matmulOp->hasAttr("aicom.linalg_tiled"))
        continue;
      auto aType =
          dyn_cast<RankedTensorType>(matmulOp.getOperand(0).getType());
      auto bType =
          dyn_cast<RankedTensorType>(matmulOp.getOperand(1).getType());
      if (!aType || !bType || aType.getRank() != 2 || bType.getRank() != 2)
        continue;
      if (aType.isDynamicDim(0) || aType.isDynamicDim(1) ||
          bType.isDynamicDim(0) || bType.isDynamicDim(1))
        continue;
      if (aType.getDimSize(0) >= kTileM * 2 && aType.getDimSize(1) >= kTileM &&
          bType.getDimSize(1) >= kTileN * 2)
        targets.push_back(matmulOp);
    }

    for (linalg::MatmulOp matmul : targets) {
      rewriter.setInsertionPoint(matmul);
      linalg::LinalgTilingOptions options;
      options.setTileSizes({kTileM, kTileM, kTileN});
      auto tiled = linalg::tileLinalgOp(
          rewriter, cast<linalg::LinalgOp>(matmul.getOperation()), options);
      if (failed(tiled))
        return signalPassFailure();
      Operation *tiledOp = tiled->op.getOperation();
      rewriter.modifyOpInPlace(tiledOp, [&] {
        tiledOp->setAttr("aicom.linalg_tiled", rewriter.getUnitAttr());
      });
    }
  }
};

} // namespace

std::unique_ptr<Pass> createCustomLinalgTilePass() {
  return std::make_unique<CustomLinalgTilePass>();
}

} // namespace aicom
} // namespace mlir
