#include "AICompiler/Pipeline.h"
#include "AICompiler/PipelineStages.h"
#include "AICompiler/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMPass.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "stablehlo/dialect/Register.h"

#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"

#include <cstdlib>

using namespace mlir;
using namespace mlir::aicom;

static bool isKnownPipelineStage(llvm::StringRef name) {
  return name == "fusion" || name == "linalg" || name == "bufferize" ||
         name == "loops" || name == "llvm" || name == "all";
}

static llvm::cl::opt<std::string> inputFilename(
    "input", llvm::cl::desc("Input StableHLO .mlir file"),
    llvm::cl::value_desc("filename"), llvm::cl::Required);

static llvm::cl::opt<bool> dumpIr("dump-ir",
                                  llvm::cl::desc("Print IR after each pass"));

static llvm::cl::opt<bool> jitRun("jit",
                                  llvm::cl::desc("JIT-compile and run entry"));

static llvm::cl::opt<bool> noVectorize(
    "no-vectorize",
    llvm::cl::desc("Use sequential linalg-to-loops instead of parallel"));

static llvm::cl::opt<std::string> entryFunc(
    "entry-func", llvm::cl::init("inference"),
    llvm::cl::desc("Function name to JIT (default: inference)"));

static llvm::cl::opt<std::string> stopAfterStage(
    "pipeline-stop-after",
    llvm::cl::desc(
        "Stop after pipeline stage: fusion, linalg, bufferize, loops, llvm, all"),
    llvm::cl::init("all"));

static void printMemRefF32(int64_t rank, void *data, int64_t offset,
                           int64_t *sizes) {
  if (rank == 0) {
    llvm::outs() << reinterpret_cast<float *>(data)[offset];
    return;
  }
  if (rank == 1) {
    auto *ptr = reinterpret_cast<float *>(data) + offset;
    for (int64_t i = 0; i < sizes[0]; ++i) {
      if (i)
        llvm::outs() << ", ";
      llvm::outs() << ptr[i];
    }
    return;
  }
  if (rank == 2) {
    auto *ptr = reinterpret_cast<float *>(data) + offset;
    for (int64_t i = 0; i < sizes[0]; ++i) {
      for (int64_t j = 0; j < sizes[1]; ++j) {
        if (i || j)
          llvm::outs() << ", ";
        llvm::outs() << ptr[i * sizes[1] + j];
      }
    }
    return;
  }
  if (rank == 4) {
    auto *ptr = reinterpret_cast<float *>(data) + offset;
    int64_t s1 = sizes[1], s2 = sizes[2], s3 = sizes[3];
    for (int64_t i0 = 0; i0 < sizes[0]; ++i0)
      for (int64_t i1 = 0; i1 < s1; ++i1)
        for (int64_t i2 = 0; i2 < s2; ++i2)
          for (int64_t i3 = 0; i3 < s3; ++i3) {
            if (i0 || i1 || i2 || i3)
              llvm::outs() << ", ";
            llvm::outs() << ptr[((i0 * s1 + i1) * s2 + i2) * s3 + i3];
          }
    return;
  }
  llvm::outs() << "<unsupported rank " << rank << " for JIT print>";
}

template <int Rank>
static LogicalResult invokeJitRanked(ExecutionEngine &engine, StringRef funcName,
                                     int64_t numElements) {
  StridedMemRefType<float, Rank> result{};
  void *args[] = {&result};
  if (auto err = engine.invokePacked(funcName, args)) {
    llvm::errs() << "JIT invocation failed: " << llvm::toString(std::move(err))
                 << "\n";
    return failure();
  }

  llvm::outs() << "JIT result (" << numElements << " elements): ";
  if constexpr (Rank == 0) {
    printMemRefF32(0, result.data, result.offset, nullptr);
  } else {
    printMemRefF32(Rank, result.data, result.offset, result.sizes);
  }
  llvm::outs() << "\n";

  if (result.basePtr)
    std::free(result.basePtr);
  return success();
}

static LogicalResult runJit(ModuleOp module, StringRef funcName) {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  mlir::registerLLVMDialectTranslation(*module->getContext());

  auto func = module.lookupSymbol<func::FuncOp>(funcName);
  if (!func) {
    llvm::errs() << "Entry function '" << funcName << "' not found\n";
    return failure();
  }
  if (func.getNumArguments() != 0) {
    llvm::errs() << "JIT demo supports only nullary functions (constants "
                    "embedded in IR).\n";
    return failure();
  }
  auto ranked = dyn_cast<RankedTensorType>(func.getFunctionType().getResult(0));
  if (!ranked || !ranked.getElementType().isF32()) {
    llvm::errs() << "JIT demo expects an f32 ranked tensor result type\n";
    return failure();
  }
  int64_t rank = ranked.getRank();
  int64_t numElements = ranked.getNumElements();

  PipelineOptions opts;
  opts.enableVectorize = !noVectorize;
  opts.dumpIrBetweenStages = dumpIr;
  if (!isKnownPipelineStage(stopAfterStage)) {
    llvm::errs() << "Unknown --pipeline-stop-after stage: "
                 << stopAfterStage.getValue()
                 << "\n";
    return failure();
  }
  opts.stopAfter = parsePipelineStage(stopAfterStage);
  if (failed(runAICompilerPipeline(module, opts)))
    return failure();

  ExecutionEngineOptions engineOptions;
  engineOptions.sharedLibPaths = {
      "/usr/local/lib/libmlir_c_runner_utils.so",
      "/usr/local/lib/libmlir_runner_utils.so",
  };
  auto engineOr = ExecutionEngine::create(module, engineOptions);
  if (!engineOr) {
    llvm::errs() << "Failed to create ExecutionEngine: "
                 << llvm::toString(engineOr.takeError()) << "\n";
    return failure();
  }
  std::unique_ptr<ExecutionEngine> engine = std::move(*engineOr);

  engine->registerSymbols([&](llvm::orc::MangleAndInterner interner) {
    llvm::orc::SymbolMap map;
    map[interner("malloc")] = {llvm::orc::ExecutorAddr::fromPtr(
                                   reinterpret_cast<void *>(::malloc)),
                               llvm::JITSymbolFlags::Exported};
    map[interner("free")] = {llvm::orc::ExecutorAddr::fromPtr(
                                 reinterpret_cast<void *>(::free)),
                             llvm::JITSymbolFlags::Exported};
    return map;
  });

  switch (rank) {
  case 0:
    return invokeJitRanked<0>(*engine, funcName, numElements);
  case 1:
    return invokeJitRanked<1>(*engine, funcName, numElements);
  case 2:
    return invokeJitRanked<2>(*engine, funcName, numElements);
  case 3:
    return invokeJitRanked<3>(*engine, funcName, numElements);
  case 4:
    return invokeJitRanked<4>(*engine, funcName, numElements);
  default:
    llvm::errs() << "JIT demo supports tensor rank <= 4, got " << rank << "\n";
    return failure();
  }
}

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);
  registerAllPasses();
  registerAICompilerPasses();

  llvm::cl::ParseCommandLineOptions(argc, argv, "MLIR AI compiler pipeline demo\n");

  DialectRegistry registry;
  registerAllDialects(registry);
  mlir::stablehlo::registerAllDialects(registry);
  registerAllExtensions(registry);
  mlir::registerConvertToLLVMDependentDialectLoading(registry);
  registerAllToLLVMIRTranslations(registry);
  registry.insert<LLVM::LLVMDialect>();

  MLIRContext context(registry);

  llvm::SourceMgr sourceMgr;
  const llvm::StringRef file = inputFilename.getValue();
  auto fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(file);
  if (!fileOrErr) {
    llvm::errs() << "Cannot open input file: " << file << "\n";
    return 1;
  }
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());

  OwningOpRef<ModuleOp> module = parseSourceFile<ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Failed to parse MLIR module\n";
    return 1;
  }

  if (jitRun) {
    if (failed(runJit(*module, entryFunc)))
      return 1;
    return 0;
  }

  PipelineOptions opts;
  opts.enableVectorize = !noVectorize;
  opts.dumpIrBetweenStages = dumpIr;
  if (!isKnownPipelineStage(stopAfterStage)) {
    llvm::errs() << "Unknown --pipeline-stop-after stage: "
                 << stopAfterStage.getValue()
                 << "\n";
    return 1;
  }
  opts.stopAfter = parsePipelineStage(stopAfterStage);
  if (failed(runAICompilerPipeline(*module, opts))) {
    llvm::errs() << "Pipeline failed\n";
    return 1;
  }

  if (!dumpIr)
    module->print(llvm::outs());
  return 0;
}
