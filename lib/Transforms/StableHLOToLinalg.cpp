#include "AICompiler/PipelineStages.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "stablehlo/conversions/linalg/transforms/Passes.h"

namespace mlir {
namespace aicom {

void buildStableHloToLinalgStage(OpPassManager &pm) {
  pm.addPass(stablehlo::createStablehloLegalizeToLinalgPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addPass(createCSEPass());
}

} // namespace aicom
} // namespace mlir
