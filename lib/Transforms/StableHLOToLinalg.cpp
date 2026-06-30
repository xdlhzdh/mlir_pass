#include "AICompiler/PipelineStages.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "stablehlo/conversions/linalg/transforms/Passes.h"
#include "stablehlo/transforms/Passes.h"

namespace mlir {
namespace aicom {

void buildStableHloToLinalgStage(OpPassManager &pm) {
  // Narrow dynamic dims when static operands imply concrete shapes (e.g. batch).
  pm.addPass(stablehlo::createStablehloRefineShapesPass());
  // GELU and other P4 exports may contain chlo.erf; legalize before Linalg.
  pm.addNestedPass<func::FuncOp>(
      stablehlo::createChloLegalizeToStablehloPass());
  pm.addPass(stablehlo::createStablehloLegalizeToLinalgPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addPass(createCSEPass());
}

} // namespace aicom
} // namespace mlir
