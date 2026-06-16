include "mlir/Pass/PassBase.td"

def ConvBNFusion : Pass<"conv-bn-fusion", "func::FuncOp"> {
  let summary = "Fuse StableHLO convolution followed by batch norm inference";
  let description = [{
    Matches `stablehlo.batch_norm_inference` whose operand is a
    `stablehlo.convolution` with constant weights and BN parameters. Folds
    BN scale into conv weights and bias (beta - mean * scale) into a
    broadcast add after the fused convolution.
  }];
  let constructor = "mlir::aicom::createConvBNFusionPass()";
  let dependentDialects = [
    "arith::ArithDialect",
    "stablehlo::StablehloDialect",
  ];
}

def CustomLinalgOpt : Pass<"custom-linalg-opt", "func::FuncOp"> {
  let summary = "Fold constant elementwise linalg.generic at compile time";
  let description = [{
    Custom teaching pass: walks `linalg.generic` with all-parallel iterators
    and constant inputs, evaluates the body (add/mul/sub/max) at compile time,
    and replaces the op with `arith.constant`.
  }];
  let constructor = "mlir::aicom::createCustomLinalgOptPass()";
  let dependentDialects = [
    "linalg::LinalgDialect",
    "arith::ArithDialect",
  ];
}

def CustomBufferOpt : Pass<"custom-buffer-opt", "func::FuncOp"> {
  let summary = "Promote small memref.alloc to memref.alloca (stack)";
  let description = [{
    Custom teaching pass: for statically-shaped allocations whose total size
    is at most 4 KiB, replace `memref.alloc` with `memref.alloca` and remove
    the paired `memref.dealloc`.
  }];
  let constructor = "mlir::aicom::createCustomBufferOptPass()";
  let dependentDialects = ["memref::MemRefDialect"];
}

def CustomLinalgToParallelLoops : Pass<"custom-linalg-to-parallel-loops", "func::FuncOp"> {
  let summary = "Lower parallel linalg ops to scf.parallel loops (teaching)";
  let description = [{
    Custom replacement for `convert-linalg-to-parallel-loops` on supported ops:
    - elementwise `linalg.generic` (all-parallel, identity maps, static memref)
      -> nested `scf.parallel` with memref load/compute/store
    - `linalg.matmul` (2D static) -> `scf.parallel` over (i,j) + `scf.for` on k
  }];
  let constructor = "mlir::aicom::createCustomLinalgToParallelLoopsPass()";
  let dependentDialects = [
    "linalg::LinalgDialect",
    "memref::MemRefDialect",
    "scf::SCFDialect",
    "arith::ArithDialect",
  ];
}

def CustomLoopTiling : Pass<"custom-loop-tiling", "func::FuncOp"> {
  let summary = "Strip-mine scf.for loops for cache-friendly tiles";
  let description = [{
    Custom teaching pass: finds outermost `scf.for` loops with constant bounds
    and trip count greater than the tile size, then splits each into an outer
    tile loop and an inner point loop (strip-mining).
  }];
  let constructor = "mlir::aicom::createCustomLoopTilingPass()";
  let dependentDialects = [
    "scf::SCFDialect",
    "arith::ArithDialect",
  ];
}

def CustomAffineOpt : Pass<"custom-affine-opt", "func::FuncOp"> {
  let summary = "Strip-mine outermost affine.for loops with constant bounds";
  let description = [{
    Custom teaching pass: finds outermost `affine.for` loops with constant
    bounds and trip count greater than the tile size, then applies
    `tilePerfectlyNested` strip-mining (tile size 4). Skips loops with
    `iter_args` (reductions).
  }];
  let constructor = "mlir::aicom::createCustomAffineOptPass()";
  let dependentDialects = ["affine::AffineDialect"];
}

def CustomVectorOpt : Pass<"custom-vector-opt", "func::FuncOp"> {
  let summary = "Lower static vector.transfer ops to vector.load/store";
  let description = [{
    Custom teaching pass: for `vector.transfer_read/write` with minor-identity
    maps and statically-shaped memrefs, replace with `vector.load/store`.
  }];
  let constructor = "mlir::aicom::createCustomVectorOptPass()";
  let dependentDialects = [
    "vector::VectorDialect",
    "memref::MemRefDialect",
  ];
}

def CustomLLVMCleanup : Pass<"custom-llvm-cleanup", "ModuleOp"> {
  let summary = "LLVM dialect dead store elimination and dead op removal";
  let description = [{
    Custom teaching pass: removes stores to `llvm.alloca` buffers that are
    never loaded, then deletes side-effect-free LLVM ops with no uses.
  }];
  let constructor = "mlir::aicom::createCustomLLVMCleanupPass()";
  let dependentDialects = ["LLVM::LLVMDialect"];
}
