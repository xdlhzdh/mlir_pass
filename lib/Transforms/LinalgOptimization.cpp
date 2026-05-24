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
  // Custom teaching pass (see CustomLinalgOpt.cpp, Passes.td: custom-linalg-opt).
  pm.addNestedPass<func::FuncOp>(createCustomLinalgOptPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addPass(createCSEPass());
}

} // namespace aicom
} // namespace mlir
