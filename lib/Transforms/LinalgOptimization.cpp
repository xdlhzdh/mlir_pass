#include "AICompiler/PipelineStages.h"

#include "AICompiler/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace aicom {

void buildLinalgOptimizationStage(OpPassManager &pm) {
  pm.addNestedPass<func::FuncOp>(createLinalgElementwiseOpFusionPass());
  pm.addNestedPass<func::FuncOp>(createLinalgFoldUnitExtentDimsPass());
  pm.addNestedPass<func::FuncOp>(createCustomLinalgOptPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  // CSE runs before tile so it does not fold the tensor tile nest away.
  pm.addNestedPass<func::FuncOp>(createCustomLinalgTilePass());
  // No second linalg-fuse-elementwise-ops here: upstream pass collapses tiled nests.
}

} // namespace aicom
} // namespace mlir
