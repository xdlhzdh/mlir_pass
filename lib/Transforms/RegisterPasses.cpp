#include "AICompiler/Passes.h"
#include "mlir/Pass/PassRegistry.h"

void registerAICompilerPasses() {
  ::mlir::registerPass([] { return mlir::aicom::createConvBNFusionPass(); });
  ::mlir::registerPass([] { return mlir::aicom::createCustomLinalgOptPass(); });
  ::mlir::registerPass([] { return mlir::aicom::createCustomBufferOptPass(); });
  ::mlir::registerPass(
      [] { return mlir::aicom::createCustomLinalgToParallelLoopsPass(); });
  ::mlir::registerPass([] { return mlir::aicom::createCustomLoopTilingPass(); });
  ::mlir::registerPass([] { return mlir::aicom::createCustomLLVMCleanupPass(); });
}
