#include "AICompiler/PipelineStages.h"

#include "AICompiler/Passes.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Pass/PassManager.h"

namespace mlir {
namespace aicom {

void buildLowerToLoopsStage(OpPassManager &pm) {
  pm.addPass(createConvertBufferizationToMemRefPass());
  pm.addPass(createConvertLinalgToLoopsPass());
  // Custom teaching pass (see CustomLoopTiling.cpp, Passes.td: custom-loop-tiling).
  pm.addNestedPass<func::FuncOp>(createCustomLoopTilingPass());
  pm.addPass(createSCFToControlFlowPass());
}

} // namespace aicom
} // namespace mlir
