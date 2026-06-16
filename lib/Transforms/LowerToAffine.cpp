#include "AICompiler/PipelineStages.h"

#include "AICompiler/Passes.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Pass/PassManager.h"

namespace mlir {
namespace aicom {

void buildAffineStage(OpPassManager &pm, bool stopAfterAffine) {
  pm.addPass(createConvertBufferizationToMemRefPass());
  pm.addPass(createConvertLinalgToAffineLoopsPass());
  pm.addNestedPass<func::FuncOp>(createCustomAffineOptPass());
  if (!stopAfterAffine) {
    pm.addPass(createLowerAffinePass());
    pm.addPass(createSCFToControlFlowPass());
  }
}

} // namespace aicom
} // namespace mlir
