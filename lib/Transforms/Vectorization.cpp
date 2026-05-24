#include "AICompiler/PipelineStages.h"

#include "AICompiler/Passes.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"

namespace mlir {
namespace aicom {

void buildVectorizationStage(OpPassManager &pm) {
  pm.addPass(createConvertBufferizationToMemRefPass());
  // Custom teaching pass (replaces official convert-linalg-to-parallel-loops).
  // See CustomLinalgToParallelLoops.cpp, Passes.td.
  pm.addNestedPass<func::FuncOp>(createCustomLinalgToParallelLoopsPass());
  pm.addPass(createSCFToControlFlowPass());
}

} // namespace aicom
} // namespace mlir
