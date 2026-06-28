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

def ConvBNReluFusion : Pass<"conv-bn-relu-fusion", "func::FuncOp"> {
  let summary = "Canonicalize StableHLO ReLU after Conv+BN fusion";
  let description = [{
    Custom teaching pass: recognizes ReLU written as
    `stablehlo.maximum(%x, zero)` after graph fusion and rewrites it to
    `stablehlo.clamp(zero, %x, +inf)`, making the activation explicit in the
    fused StableHLO stage.
  }];
  let constructor = "mlir::aicom::createConvBNReluFusionPass()";
  let dependentDialects = [
    "stablehlo::StablehloDialect",
  ];
}

def SoftmaxLegalize : Pass<"softmax-legalize", "func::FuncOp"> {
  let summary = "Canonicalize StableHLO softmax decomposition";
  let description = [{
    Custom teaching pass: recognizes the numerically stable StableHLO softmax
    decomposition `exp(x - reduce_max(x)) / reduce_sum(exp(...))` and annotates
    the final divide so later pipeline stages and tests can identify the
    softmax boundary explicitly.
  }];
  let constructor = "mlir::aicom::createSoftmaxLegalizePass()";
  let dependentDialects = [
    "stablehlo::StablehloDialect",
  ];
}

def RMSNormLegalize : Pass<"rmsnorm-legalize", "func::FuncOp"> {
  let summary = "Canonicalize StableHLO RMSNorm decomposition";
  let description = [{
    Recognizes RMSNorm decomposed as `x / sqrt(reduce_mean(x*x) + eps) * weight`
    and annotates the final multiply with `aicom.rmsnorm_canonicalized`.
  }];
  let constructor = "mlir::aicom::createRMSNormLegalizePass()";
  let dependentDialects = [
    "stablehlo::StablehloDialect",
  ];
}

def AttentionLegalize : Pass<"attention-legalize", "func::FuncOp"> {
  let summary = "Mark scaled dot-product attention subgraph boundary";
  let description = [{
    After `softmax-legalize`, recognizes `dot_general(softmax(...), V)` as the
    second matmul in scaled dot-product attention and annotates it with
    `aicom.scaled_dot_product_attention`.
  }];
  let constructor = "mlir::aicom::createAttentionLegalizePass()";
  let dependentDialects = [
    "stablehlo::StablehloDialect",
  ];
}

def RoPELegalize : Pass<"rope-legalize", "func::FuncOp"> {
  let summary = "Canonicalize StableHLO RoPE decomposition";
  let description = [{
    Recognizes rotate-half style RoPE `x*cos + rotate_half(x)*sin` and annotates
    the final add with `aicom.rope_canonicalized`.
  }];
  let constructor = "mlir::aicom::createRoPELegalizePass()";
  let dependentDialects = [
    "stablehlo::StablehloDialect",
  ];
}

def LayerNormLegalize : Pass<"layernorm-legalize", "func::FuncOp"> {
  let summary = "Canonicalize StableHLO LayerNorm decomposition";
  let description = [{
    Recognizes LayerNorm `gamma * (x - mean) / sqrt(var + eps) + beta` and annotates
    the final add with `aicom.layernorm_canonicalized`.
  }];
  let constructor = "mlir::aicom::createLayerNormLegalizePass()";
  let dependentDialects = [
    "stablehlo::StablehloDialect",
  ];
}

def StablehloConstantFold : Pass<"stablehlo-constant-fold", "func::FuncOp"> {
  let summary = "Fold constant StableHLO elementwise add/mul at compile time";
  let description = [{
    Custom teaching pass: when both operands of `stablehlo.add` or
    `stablehlo.multiply` are compile-time constants, evaluate the op and
    replace it with a single `stablehlo.constant`.
  }];
  let constructor = "mlir::aicom::createStablehloConstantFoldPass()";
  let dependentDialects = [
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
