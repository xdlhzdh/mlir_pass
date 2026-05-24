#include "AICompiler/Pipeline.h"
#include "AICompiler/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"

namespace mlir {
namespace aicom {

namespace {

bool runThroughStage(PipelineStage stage, PipelineStage stopAfter) {
  if (stopAfter == PipelineStage::All)
    return true;
  return static_cast<int>(stage) <= static_cast<int>(stopAfter);
}

LogicalResult runOneStage(ModuleOp module, const PipelineOptions &options,
                          PipelineStage stage, llvm::StringRef label,
                          llvm::function_ref<void(OpPassManager &)> build) {
  PassManager pm(module.getContext());
  pm.enableVerifier(true);
  if (options.dumpIrBetweenStages) {
    OpPrintingFlags flags;
    flags.enableDebugInfo(false, false);
    pm.getContext()->disableMultithreading();
    pm.enableIRPrinting(
        [](Pass *, Operation *) { return false; },
        [](Pass *, Operation *) { return true; }, true, false, false,
        llvm::errs(), flags);
  }
  build(pm);
  if (failed(pm.run(module)))
    return failure();
  if (options.dumpIrBetweenStages) {
    llvm::errs() << "\n// -----// IR Dump After Stage: " << label
                 << " //----- //\n";
    module.print(llvm::errs());
    llvm::errs() << "\n";
  }
  return success();
}

} // namespace

void buildAICompilerPipeline(OpPassManager &pm, const PipelineOptions &options) {
  if (runThroughStage(PipelineStage::Fusion, options.stopAfter))
    buildFusionStage(pm);
  if (runThroughStage(PipelineStage::Linalg, options.stopAfter)) {
    buildStableHloToLinalgStage(pm);
    buildLinalgOptimizationStage(pm);
  }
  if (runThroughStage(PipelineStage::Bufferize, options.stopAfter))
    buildBufferizationStage(pm);
  if (runThroughStage(PipelineStage::Loops, options.stopAfter)) {
    if (options.enableVectorize)
      buildVectorizationStage(pm);
    else
      buildLowerToLoopsStage(pm);
  }
  if (runThroughStage(PipelineStage::LLVM, options.stopAfter))
    buildLLVMLoweringStage(pm);
}

LogicalResult runAICompilerPipeline(ModuleOp module,
                                    const PipelineOptions &options) {
  auto runIf = [&](PipelineStage stage, llvm::StringRef label,
                   llvm::function_ref<void(OpPassManager &)> build) {
    if (!runThroughStage(stage, options.stopAfter))
      return success();
    return runOneStage(module, options, stage, label, build);
  };

  if (failed(runIf(PipelineStage::Fusion, "fusion",
                   [](OpPassManager &pm) { buildFusionStage(pm); })))
    return failure();

  if (failed(runIf(PipelineStage::Linalg, "linalg", [](OpPassManager &pm) {
        buildStableHloToLinalgStage(pm);
        buildLinalgOptimizationStage(pm);
      })))
    return failure();

  if (failed(runIf(PipelineStage::Bufferize, "bufferize",
                   [](OpPassManager &pm) { buildBufferizationStage(pm); })))
    return failure();

  if (failed(runIf(PipelineStage::Loops, "loops", [&](OpPassManager &pm) {
        if (options.enableVectorize)
          buildVectorizationStage(pm);
        else
          buildLowerToLoopsStage(pm);
      })))
    return failure();

  if (failed(runIf(PipelineStage::LLVM, "llvm",
                   [](OpPassManager &pm) { buildLLVMLoweringStage(pm); })))
    return failure();

  return success();
}

} // namespace aicom
} // namespace mlir
