// AICompilerPlugin.cpp — mlir-opt pass/dialect plugin (fusion stage only).
#include "AICompiler/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/Plugins/DialectPlugin.h"
#include "mlir/Tools/Plugins/PassPlugin.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace {

void registerFusionPasses() {
  ::mlir::registerPass([] { return mlir::aicom::createConvBNFusionPass(); });
  ::mlir::registerPass(
      [] { return mlir::aicom::createConvBNReluFusionPass(); });
  ::mlir::registerPass([] { return mlir::aicom::createSoftmaxLegalizePass(); });
  ::mlir::registerPass([] { return mlir::aicom::createRMSNormLegalizePass(); });
  ::mlir::registerPass([] { return mlir::aicom::createAttentionLegalizePass(); });
  ::mlir::registerPass([] { return mlir::aicom::createFlashAttentionTilePass(); });
  ::mlir::registerPass([] { return mlir::aicom::createRoPELegalizePass(); });
  ::mlir::registerPass([] { return mlir::aicom::createLayerNormLegalizePass(); });
  ::mlir::registerPass([] { return mlir::aicom::createGeluLegalizePass(); });
  ::mlir::registerPass([] { return mlir::aicom::createSwiGLULegalizePass(); });
  ::mlir::registerPass([] { return mlir::aicom::createQdqLegalizePass(); });
  ::mlir::registerPass([] { return mlir::aicom::createMatMulBiasFusionPass(); });
  ::mlir::registerPass(
      [] { return mlir::aicom::createHorizontalGemmFusionPass(); });
  ::mlir::registerPass(
      [] { return mlir::aicom::createElementwiseChainLegalizePass(); });
  ::mlir::registerPass(
      [] { return mlir::aicom::createProducerConsumerLegalizePass(); });
  ::mlir::registerPass(
      [] { return mlir::aicom::createLayoutBridgeLegalizePass(); });
  ::mlir::registerPass(
      [] { return mlir::aicom::createKVCacheLegalizePass(); });
  ::mlir::registerPass(
      [] { return mlir::aicom::createBroadcastSimplifyPass(); });
  ::mlir::registerPass(
      [] { return mlir::aicom::createStablehloConstantFoldPass(); });
}

void registerFusionPipeline() {
  ::mlir::registerPassPipeline(
      "aicom-fusion", "mlir_pass StableHLO fusion stage passes",
      [](::mlir::OpPassManager &pm, llvm::StringRef options,
         ::mlir::function_ref<::mlir::LogicalResult(const llvm::Twine &)>
             errorHandler) {
        if (!options.empty())
          return errorHandler("no options expected");
        pm.addNestedPass<::mlir::func::FuncOp>(
            mlir::aicom::createConvBNFusionPass());
        pm.addNestedPass<::mlir::func::FuncOp>(
            mlir::aicom::createConvBNReluFusionPass());
        pm.addNestedPass<::mlir::func::FuncOp>(
            mlir::aicom::createSoftmaxLegalizePass());
        pm.addNestedPass<::mlir::func::FuncOp>(
            mlir::aicom::createRMSNormLegalizePass());
        pm.addNestedPass<::mlir::func::FuncOp>(
            mlir::aicom::createAttentionLegalizePass());
        pm.addNestedPass<::mlir::func::FuncOp>(
            mlir::aicom::createFlashAttentionTilePass());
        pm.addNestedPass<::mlir::func::FuncOp>(
            mlir::aicom::createRoPELegalizePass());
        pm.addNestedPass<::mlir::func::FuncOp>(
            mlir::aicom::createLayerNormLegalizePass());
        pm.addNestedPass<::mlir::func::FuncOp>(
            mlir::aicom::createGeluLegalizePass());
        pm.addNestedPass<::mlir::func::FuncOp>(
            mlir::aicom::createSwiGLULegalizePass());
        pm.addNestedPass<::mlir::func::FuncOp>(
            mlir::aicom::createQdqLegalizePass());
        pm.addNestedPass<::mlir::func::FuncOp>(
            mlir::aicom::createMatMulBiasFusionPass());
        pm.addNestedPass<::mlir::func::FuncOp>(
            mlir::aicom::createHorizontalGemmFusionPass());
        pm.addNestedPass<::mlir::func::FuncOp>(
            mlir::aicom::createElementwiseChainLegalizePass());
        pm.addNestedPass<::mlir::func::FuncOp>(
            mlir::aicom::createProducerConsumerLegalizePass());
        pm.addNestedPass<::mlir::func::FuncOp>(
            mlir::aicom::createLayoutBridgeLegalizePass());
        pm.addNestedPass<::mlir::func::FuncOp>(
            mlir::aicom::createLayoutFoldPass());
        pm.addNestedPass<::mlir::func::FuncOp>(
            mlir::aicom::createKVCacheLegalizePass());
        pm.addNestedPass<::mlir::func::FuncOp>(
            mlir::aicom::createBroadcastSimplifyPass());
        pm.addNestedPass<::mlir::func::FuncOp>(
            mlir::aicom::createStablehloConstantFoldPass());
        return ::mlir::success();
      },
      [](::mlir::function_ref<void(const ::mlir::detail::PassOptions &)>){});
}

} // namespace

extern "C" LLVM_ATTRIBUTE_WEAK ::mlir::PassPluginLibraryInfo
mlirGetPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "AICompilerPasses", "v0.1", []() {
            registerFusionPasses();
            registerFusionPipeline();
          }};
}

extern "C" LLVM_ATTRIBUTE_WEAK ::mlir::DialectPluginLibraryInfo
mlirGetDialectPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "AICompilerStableHLO", "v0.1",
          [](::mlir::DialectRegistry *registry) {
            if (registry)
              registry->insert<::mlir::stablehlo::StablehloDialect,
                               ::mlir::arith::ArithDialect,
                               ::mlir::func::FuncDialect>();
          }};
}
