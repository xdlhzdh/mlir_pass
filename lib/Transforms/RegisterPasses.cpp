#include "AICompiler/Passes.h"

#include "mlir/Pass/PassRegistry.h"
#include "llvm/Support/raw_ostream.h"

void registerAICompilerPasses() {
  ::mlir::registerPass([] { return mlir::aicom::createConvBNFusionPass(); });
  ::mlir::registerPass([] { return mlir::aicom::createConvBNReluFusionPass(); });
  ::mlir::registerPass([] { return mlir::aicom::createSoftmaxLegalizePass(); });
  ::mlir::registerPass([] { return mlir::aicom::createCustomLinalgOptPass(); });
  ::mlir::registerPass([] { return mlir::aicom::createCustomBufferOptPass(); });
  ::mlir::registerPass(
      [] { return mlir::aicom::createCustomLinalgToParallelLoopsPass(); });
  ::mlir::registerPass([] { return mlir::aicom::createCustomLoopTilingPass(); });
  ::mlir::registerPass([] { return mlir::aicom::createCustomAffineOptPass(); });
  ::mlir::registerPass([] { return mlir::aicom::createCustomVectorOptPass(); });
  ::mlir::registerPass([] { return mlir::aicom::createCustomLLVMCleanupPass(); });
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
      "custom-linalg-opt",               // linalg
      "custom-buffer-opt",               // bufferize
      "custom-loop-tiling",              // loops / scf-seq
      "custom-linalg-to-parallel-loops", // loops / scf-par
      "custom-affine-opt",               // affine
      "custom-vector-opt",               // vector
      "custom-llvm-cleanup",             // llvm
  };
  static constexpr llvm::StringRef kStages[] = {
      "fusion", "fusion", "fusion", "linalg", "bufferize",
      "loops/scf-seq", "loops/scf-par", "affine", "vector", "llvm",
  };
  static constexpr size_t kNumPasses = sizeof(kPasses) / sizeof(kPasses[0]);
  for (size_t i = 0; i < kNumPasses; ++i)
    os << kPasses[i] << "  [stage: " << kStages[i] << "]\n";
}

} // namespace aicom
} // namespace mlir
