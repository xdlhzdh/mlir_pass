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
  ::mlir::registerPass(
      [] { return mlir::aicom::createStablehloConstantFoldPass(); });
  ::mlir::registerPass([] { return mlir::aicom::createConvBNFusionPass(); });
  ::mlir::registerPass(
      [] { return mlir::aicom::createConvBNReluFusionPass(); });
  ::mlir::registerPass([] { return mlir::aicom::createSoftmaxLegalizePass(); });
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
            mlir::aicom::createStablehloConstantFoldPass());
        pm.addNestedPass<::mlir::func::FuncOp>(
            mlir::aicom::createConvBNFusionPass());
        pm.addNestedPass<::mlir::func::FuncOp>(
            mlir::aicom::createConvBNReluFusionPass());
        pm.addNestedPass<::mlir::func::FuncOp>(
            mlir::aicom::createSoftmaxLegalizePass());
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
