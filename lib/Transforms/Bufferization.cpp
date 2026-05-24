#include "AICompiler/PipelineStages.h"

#include "AICompiler/Passes.h"

#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace aicom {

void buildBufferizationStage(OpPassManager &pm) {
  bufferization::OneShotBufferizePassOptions bufferizeOpts;
  bufferizeOpts.bufferizeFunctionBoundaries = true;
  pm.addPass(bufferization::createOneShotBufferizePass(bufferizeOpts));
  bufferization::buildBufferDeallocationPipeline(pm);
  // Custom teaching pass (see CustomBufferOpt.cpp, Passes.td: custom-buffer-opt).
  pm.addNestedPass<func::FuncOp>(createCustomBufferOptPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addPass(createCSEPass());
}

} // namespace aicom
} // namespace mlir
