#include "AICompiler/PipelineStages.h"

#include "AICompiler/Passes.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"

namespace mlir {
namespace aicom {

void buildLLVMLoweringStage(OpPassManager &pm, LoopMode loopMode) {
  // expand_shape / collapse_shape / subview have no direct LLVM lowering; the
  // affine.apply ops this emits are lowered right after.
  pm.addPass(memref::createExpandStridedMetadataPass());
  pm.addPass(createLowerAffinePass());
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(createConvertControlFlowToLLVMPass());
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createConvertIndexToLLVMPass());
  pm.addPass(createConvertMathToLLVMPass());
  if (loopMode == LoopMode::Vector)
    pm.addPass(createConvertVectorToLLVMPass());
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
  pm.addPass(createCustomLLVMCleanupPass());
}

} // namespace aicom
} // namespace mlir
