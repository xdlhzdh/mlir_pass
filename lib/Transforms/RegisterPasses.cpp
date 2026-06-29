#include "AICompiler/Passes.h"
#include "AICompiler/PipelineStages.h"

#include "mlir/Pass/PassRegistry.h"
#include "llvm/Support/raw_ostream.h"

void registerAICompilerPasses() {
  ::mlir::registerPass([] { return mlir::aicom::createConvBNFusionPass(); });
  ::mlir::registerPass([] { return mlir::aicom::createConvBNReluFusionPass(); });
  ::mlir::registerPass([] { return mlir::aicom::createSoftmaxLegalizePass(); });
  ::mlir::registerPass([] { return mlir::aicom::createRMSNormLegalizePass(); });
  ::mlir::registerPass([] { return mlir::aicom::createAttentionLegalizePass(); });
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
      [] { return mlir::aicom::createStablehloConstantFoldPass(); });
  ::mlir::registerPass([] { return mlir::aicom::createCustomLinalgOptPass(); });
  ::mlir::registerPass([] { return mlir::aicom::createCustomBufferOptPass(); });
  ::mlir::registerPass(
      [] { return mlir::aicom::createCustomLinalgToParallelLoopsPass(); });
  ::mlir::registerPass([] { return mlir::aicom::createCustomLoopTilingPass(); });
  ::mlir::registerPass([] { return mlir::aicom::createCustomAffineOptPass(); });
  ::mlir::registerPass([] { return mlir::aicom::createCustomVectorOptPass(); });
  ::mlir::registerPass([] { return mlir::aicom::createCustomLLVMCleanupPass(); });
}

void registerAICompilerPassPipelines() {
  ::mlir::registerPassPipeline(
      "aicom-fusion", "mlir_pass StableHLO fusion stage passes",
      [](::mlir::OpPassManager &pm, llvm::StringRef options,
         ::mlir::function_ref<::mlir::LogicalResult(const llvm::Twine &)>
             errorHandler) {
        if (!options.empty())
          return errorHandler("no options expected");
        mlir::aicom::buildFusionStage(pm);
        return ::mlir::success();
      },
      [](::mlir::function_ref<void(const ::mlir::detail::PassOptions &)>){});
}

namespace mlir {
namespace aicom {

void printAICompilerPassList(llvm::raw_ostream &os) {
  os << "== loop-mode paths (--loop-mode) ==\n";
  os << "scf-seq    -> custom-loop-tiling"
        "  (+ convert-linalg-to-loops, convert-scf-to-cf)\n";
  os << "scf-par    -> custom-linalg-to-parallel-loops"
        "  (+ convert-scf-to-cf)\n";
  os << "affine     -> custom-affine-opt"
        "  (+ convert-linalg-to-affine-loops, lower-affine, convert-scf-to-cf)\n";
  os << "vector     -> custom-vector-opt"
        "  (+ convert-linalg-to-affine-loops, affine-super-vectorize, "
        "lower-affine, convert-scf-to-cf, convert-vector-to-llvm)\n";
  os << "\n== custom teaching passes ==\n";

  static constexpr llvm::StringRef kPasses[] = {
      "conv-bn-fusion",                  // fusion
      "conv-bn-relu-fusion",             // fusion
      "softmax-legalize",                // fusion
      "rmsnorm-legalize",                // fusion
      "attention-legalize",              // fusion
      "rope-legalize",                   // fusion
      "layernorm-legalize",              // fusion
      "gelu-legalize",                   // fusion
      "swiglu-legalize",                 // fusion
      "qdq-legalize",                    // fusion
      "matmul-bias-fusion",              // fusion
      "horizontal-gemm-fusion",          // fusion
      "elementwise-chain-legalize",      // fusion
      "producer-consumer-legalize",      // fusion
      "layout-bridge-legalize",          // fusion
      "kvcache-legalize",                // fusion
      "stablehlo-constant-fold",         // fusion
      "custom-linalg-opt",               // linalg
      "custom-buffer-opt",               // bufferize
      "custom-loop-tiling",              // loops / scf-seq
      "custom-linalg-to-parallel-loops", // loops / scf-par
      "custom-affine-opt",               // affine
      "custom-vector-opt",               // vector
      "custom-llvm-cleanup",             // llvm
  };
  static constexpr llvm::StringRef kStages[] = {
      "fusion", "fusion", "fusion", "fusion", "fusion", "fusion", "fusion",
      "fusion", "fusion", "fusion", "fusion", "fusion", "fusion", "fusion",
      "fusion", "fusion", "linalg", "bufferize", "loops/scf-seq", "loops/scf-par",
      "affine", "vector", "llvm",
  };
  static constexpr size_t kNumPasses = sizeof(kPasses) / sizeof(kPasses[0]);
  for (size_t i = 0; i < kNumPasses; ++i)
    os << kPasses[i] << "  [stage: " << kStages[i] << "]\n";
}

} // namespace aicom
} // namespace mlir
