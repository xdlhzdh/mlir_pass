#ifndef AICOMPILER_PASSES_H
#define AICOMPILER_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace aicom {

#define GEN_PASS_DECL
#include "Passes.h.inc"

std::unique_ptr<Pass> createConvBNFusionPass();
std::unique_ptr<Pass> createCustomLinalgOptPass();
std::unique_ptr<Pass> createCustomBufferOptPass();
std::unique_ptr<Pass> createCustomLinalgToParallelLoopsPass();
std::unique_ptr<Pass> createCustomLoopTilingPass();
std::unique_ptr<Pass> createCustomAffineOptPass();
std::unique_ptr<Pass> createCustomVectorOptPass();
std::unique_ptr<Pass> createCustomLLVMCleanupPass();

} // namespace aicom
} // namespace mlir

void registerAICompilerPasses();

#endif // AICOMPILER_PASSES_H
