#include "AICompiler/PipelineStages.h"
#include "AICompiler/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/StringSwitch.h"

namespace mlir {
namespace aicom {

PipelineStage parsePipelineStage(llvm::StringRef name) {
  return llvm::StringSwitch<PipelineStage>(name)
      .Case("fusion", PipelineStage::Fusion)
      .Case("linalg", PipelineStage::Linalg)
      .Case("bufferize", PipelineStage::Bufferize)
      .Case("loops", PipelineStage::Loops)
      .Case("affine", PipelineStage::Affine)
      .Case("vector", PipelineStage::Vector)
      .Case("llvm", PipelineStage::LLVM)
      .Case("all", PipelineStage::All)
      .Default(PipelineStage::All);
}

llvm::StringRef pipelineStageName(PipelineStage stage) {
  switch (stage) {
  case PipelineStage::Fusion:
    return "fusion";
  case PipelineStage::Linalg:
    return "linalg";
  case PipelineStage::Bufferize:
    return "bufferize";
  case PipelineStage::Loops:
    return "loops";
  case PipelineStage::Affine:
    return "affine";
  case PipelineStage::Vector:
    return "vector";
  case PipelineStage::LLVM:
    return "llvm";
  case PipelineStage::All:
    return "all";
  }
  return "unknown";
}

LoopMode parseLoopMode(llvm::StringRef name) {
  return llvm::StringSwitch<LoopMode>(name)
      .Case("scf-seq", LoopMode::ScfSeq)
      .Case("scf-par", LoopMode::ScfPar)
      .Case("affine", LoopMode::Affine)
      .Case("vector", LoopMode::Vector)
      .Default(LoopMode::ScfPar);
}

llvm::StringRef loopModeName(LoopMode mode) {
  switch (mode) {
  case LoopMode::ScfSeq:
    return "scf-seq";
  case LoopMode::ScfPar:
    return "scf-par";
  case LoopMode::Affine:
    return "affine";
  case LoopMode::Vector:
    return "vector";
  }
  return "unknown";
}

LogicalResult validatePipelineOptions(PipelineStage stopAfter,
                                      LoopMode loopMode) {
  if (stopAfter == PipelineStage::All || stopAfter == PipelineStage::Fusion ||
      stopAfter == PipelineStage::Linalg ||
      stopAfter == PipelineStage::Bufferize || stopAfter == PipelineStage::LLVM)
    return success();

  if (stopAfter == PipelineStage::Loops &&
      (loopMode == LoopMode::ScfSeq || loopMode == LoopMode::ScfPar))
    return success();
  if (stopAfter == PipelineStage::Affine && loopMode == LoopMode::Affine)
    return success();
  if (stopAfter == PipelineStage::Vector && loopMode == LoopMode::Vector)
    return success();

  return failure();
}

void buildFusionStage(OpPassManager &pm) {
  pm.addNestedPass<func::FuncOp>(createConvBNFusionPass());
  pm.addNestedPass<func::FuncOp>(createConvBNReluFusionPass());
  pm.addNestedPass<func::FuncOp>(createSoftmaxLegalizePass());
  pm.addNestedPass<func::FuncOp>(createRMSNormLegalizePass());
  pm.addNestedPass<func::FuncOp>(createAttentionLegalizePass());
  pm.addNestedPass<func::FuncOp>(createRoPELegalizePass());
  pm.addNestedPass<func::FuncOp>(createLayerNormLegalizePass());
  pm.addNestedPass<func::FuncOp>(createStablehloConstantFoldPass());
}

} // namespace aicom
} // namespace mlir
