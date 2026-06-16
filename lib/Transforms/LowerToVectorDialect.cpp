#include "AICompiler/PipelineStages.h"

#include "AICompiler/Passes.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Pass/PassManager.h"

namespace mlir {
namespace aicom {

void buildVectorDialectStage(OpPassManager &pm, bool stopAfterVector) {
  pm.addPass(createConvertBufferizationToMemRefPass());
  pm.addPass(createConvertLinalgToAffineLoopsPass());

  affine::AffineVectorizeOptions vectorizeOpts;
  vectorizeOpts.vectorSizes = {2};
  vectorizeOpts.vectorizeReductions = false;
  pm.addNestedPass<func::FuncOp>(affine::createAffineVectorize(vectorizeOpts));

  pm.addNestedPass<func::FuncOp>(createCustomVectorOptPass());
  if (!stopAfterVector) {
    pm.addPass(createLowerAffinePass());
    pm.addPass(createSCFToControlFlowPass());
  }
}

} // namespace aicom
} // namespace mlir
