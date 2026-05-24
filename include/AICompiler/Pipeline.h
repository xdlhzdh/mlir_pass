#ifndef AICOMPILER_PIPELINE_H
#define AICOMPILER_PIPELINE_H

#include "AICompiler/PipelineStages.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace aicom {

struct PipelineOptions {
  bool enableVectorize = true;
  bool dumpIrBetweenStages = false;
  /// Run through this stage (inclusive). `All` runs the full pipeline.
  PipelineStage stopAfter = PipelineStage::All;
};

void buildAICompilerPipeline(OpPassManager &pm, const PipelineOptions &options);

LogicalResult runAICompilerPipeline(ModuleOp module, const PipelineOptions &options);

} // namespace aicom
} // namespace mlir

#endif // AICOMPILER_PIPELINE_H
