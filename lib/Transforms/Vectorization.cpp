#include "AICompiler/PipelineStages.h"

#include "AICompiler/Passes.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Pass/PassManager.h"

namespace mlir {
namespace aicom {

void buildVectorizationStage(OpPassManager &pm) {
  pm.addPass(createConvertBufferizationToMemRefPass());
  // Custom patterns (elementwise generic, fill, 2D matmul) go first; the
  // upstream pass lowers any remaining linalg op so none reaches LLVM lowering.
  pm.addNestedPass<func::FuncOp>(createCustomLinalgToParallelLoopsPass());
  pm.addNestedPass<func::FuncOp>(createConvertLinalgToParallelLoopsPass());
  pm.addPass(createSCFToControlFlowPass());
}

} // namespace aicom
} // namespace mlir
