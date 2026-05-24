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
  case PipelineStage::LLVM:
    return "llvm";
  case PipelineStage::All:
    return "all";
  }
  return "unknown";
}

void buildFusionStage(OpPassManager &pm) {
  pm.addNestedPass<func::FuncOp>(createConvBNFusionPass());
}

} // namespace aicom
} // namespace mlir
