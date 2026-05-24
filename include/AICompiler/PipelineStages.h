#ifndef AICOMPILER_PIPELINESTAGES_H
#define AICOMPILER_PIPELINESTAGES_H

#include "AICompiler/Passes.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Pass/PassManager.h"

namespace mlir {
namespace aicom {

/// Named pipeline stages (matches design spec section 2).
enum class PipelineStage {
  Fusion = 0,
  Linalg,
  Bufferize,
  Loops,
  LLVM,
  All,
};

/// Parse CLI stage name; returns All on unknown (caller should validate).
PipelineStage parsePipelineStage(llvm::StringRef name);

llvm::StringRef pipelineStageName(PipelineStage stage);

void buildFusionStage(OpPassManager &pm);
void buildStableHloToLinalgStage(OpPassManager &pm);
void buildLinalgOptimizationStage(OpPassManager &pm);
void buildBufferizationStage(OpPassManager &pm);
void buildLowerToLoopsStage(OpPassManager &pm);
void buildVectorizationStage(OpPassManager &pm);
void buildLLVMLoweringStage(OpPassManager &pm);

} // namespace aicom
} // namespace mlir

#endif // AICOMPILER_PIPELINESTAGES_H
