#include "AICompiler/Passes.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Operation.h"

namespace mlir {
namespace aicom {

#define GEN_PASS_DEF_CUSTOMLLVMCLEANUP
#include "Passes.h.inc"

namespace {

struct CustomLLVMCleanupPass
    : public impl::CustomLLVMCleanupBase<CustomLLVMCleanupPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    eliminateDeadStores(module);
    eliminateDeadOps(module);
  }

  static void eliminateDeadStores(ModuleOp module) {
    SmallVector<LLVM::StoreOp> deadStores;
    module.walk([&](LLVM::AllocaOp allocaOp) {
      bool hasLoad = false;
      bool hasStore = false;
      SmallVector<LLVM::StoreOp> stores;

      for (Operation *user : allocaOp.getResult().getUsers()) {
        if (isa<LLVM::LoadOp>(user)) {
          hasLoad = true;
        } else if (auto store = dyn_cast<LLVM::StoreOp>(user)) {
          hasStore = true;
          stores.push_back(store);
        } else if (isa<LLVM::GEPOp>(user)) {
          hasLoad = true;
        } else {
          hasLoad = true;
        }
      }

      if (hasStore && !hasLoad)
        deadStores.append(stores.begin(), stores.end());
    });

    for (auto store : deadStores)
      store->erase();
  }

  static void eliminateDeadOps(ModuleOp module) {
    bool changed = true;
    while (changed) {
      changed = false;
      SmallVector<Operation *> deadOps;
      module.walk([&](Operation *op) {
        if (op->getNumResults() == 0)
          return;
        if (!isMemoryEffectFree(op))
          return;
        if (op->use_empty()) {
          deadOps.push_back(op);
          changed = true;
        }
      });
      for (Operation *op : llvm::reverse(deadOps))
        op->erase();
    }
  }
};

} // namespace

std::unique_ptr<Pass> createCustomLLVMCleanupPass() {
  return std::make_unique<CustomLLVMCleanupPass>();
}

} // namespace aicom
} // namespace mlir
