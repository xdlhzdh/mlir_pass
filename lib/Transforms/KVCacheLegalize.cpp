#include "AICompiler/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace aicom {

#define GEN_PASS_DEF_KVCACHELEGALIZE
#include "Passes.h.inc"

struct KVCacheLegalizePass
    : public impl::KVCacheLegalizeBase<KVCacheLegalizePass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func->hasAttr("aicom.kvcache_boundary"))
      return;
    auto kvRole = func->getAttrOfType<StringAttr>("aicom.kv_role");
    if (!kvRole)
      return;
    if (kvRole.getValue() != "K" && kvRole.getValue() != "V")
      return;
    func->setAttr("aicom.kvcache_boundary", UnitAttr::get(&getContext()));
  }
};

std::unique_ptr<Pass> createKVCacheLegalizePass() {
  return std::make_unique<KVCacheLegalizePass>();
}

} // namespace aicom
} // namespace mlir
