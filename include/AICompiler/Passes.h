#ifndef AICOMPILER_PASSES_H
#define AICOMPILER_PASSES_H

#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace aicom {

#define GEN_PASS_DECL
#include "Passes.h.inc"

std::unique_ptr<Pass> createConvBNFusionPass();
std::unique_ptr<Pass> createConvBNReluFusionPass();
std::unique_ptr<Pass> createSoftmaxLegalizePass();
std::unique_ptr<Pass> createStablehloConstantFoldPass();
std::unique_ptr<Pass> createCustomLinalgOptPass();
std::unique_ptr<Pass> createCustomBufferOptPass();
std::unique_ptr<Pass> createCustomLinalgToParallelLoopsPass();
std::unique_ptr<Pass> createCustomLoopTilingPass();
std::unique_ptr<Pass> createCustomAffineOptPass();
std::unique_ptr<Pass> createCustomVectorOptPass();
std::unique_ptr<Pass> createCustomLLVMCleanupPass();

void printAICompilerPassList(llvm::raw_ostream &os);

} // namespace aicom
} // namespace mlir

void registerAICompilerPasses();
void registerAICompilerPassPipelines();

#endif // AICOMPILER_PASSES_H
