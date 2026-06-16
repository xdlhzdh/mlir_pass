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

void buildLoopModeStage(OpPassManager &pm, const PipelineOptions &options) {
  switch (options.loopMode) {
  case LoopMode::ScfSeq:
    buildLowerToLoopsStage(pm);
    break;
  case LoopMode::ScfPar:
    buildVectorizationStage(pm);
    break;
  case LoopMode::Affine:
    buildAffineStage(pm, options.stopAfter == PipelineStage::Affine);
    break;
  case LoopMode::Vector:
    buildVectorDialectStage(pm, options.stopAfter == PipelineStage::Vector);
    break;
  }
}

PipelineStage loopModeStage(LoopMode mode) {
  switch (mode) {
  case LoopMode::ScfSeq:
  case LoopMode::ScfPar:
    return PipelineStage::Loops;
  case LoopMode::Affine:
    return PipelineStage::Affine;
  case LoopMode::Vector:
    return PipelineStage::Vector;
  }
  return PipelineStage::Loops;
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

  PipelineStage modeStage = loopModeStage(options.loopMode);
  if (runThroughStage(modeStage, options.stopAfter))
    buildLoopModeStage(pm, options);

  if (runThroughStage(PipelineStage::LLVM, options.stopAfter))
    buildLLVMLoweringStage(pm, options.loopMode);
}

LogicalResult runAICompilerPipeline(ModuleOp module,
                                    const PipelineOptions &options) {
  if (failed(validatePipelineOptions(options.stopAfter, options.loopMode))) {
    llvm::errs() << "Invalid --pipeline-stop-after="
                 << pipelineStageName(options.stopAfter)
                 << " for --loop-mode=" << loopModeName(options.loopMode)
                 << "\n";
    return failure();
  }

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

  PipelineStage modeStage = loopModeStage(options.loopMode);
  llvm::StringRef modeLabel = pipelineStageName(modeStage);
  if (failed(runIf(modeStage, modeLabel, [&](OpPassManager &pm) {
        buildLoopModeStage(pm, options);
      })))
    return failure();

  if (failed(runIf(PipelineStage::LLVM, "llvm", [&](OpPassManager &pm) {
        buildLLVMLoweringStage(pm, options.loopMode);
      })))
    return failure();

  return success();
}

} // namespace aicom
} // namespace mlir
