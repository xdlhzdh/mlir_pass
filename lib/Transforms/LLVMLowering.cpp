#include "AICompiler/PipelineStages.h"

#include "AICompiler/Passes.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Pass/PassManager.h"

namespace mlir {
namespace aicom {

void buildLLVMLoweringStage(OpPassManager &pm) {
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(createConvertControlFlowToLLVMPass());
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createConvertIndexToLLVMPass());
  pm.addPass(createConvertMathToLLVMPass());
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
  // Custom teaching pass (see CustomLLVMCleanup.cpp, Passes.td: custom-llvm-cleanup).
  pm.addPass(createCustomLLVMCleanupPass());
}

} // namespace aicom
} // namespace mlir
