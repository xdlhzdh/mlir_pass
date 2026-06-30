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

def FlashAttentionTile : Pass<"flash-attention-tile", "func::FuncOp"> {
  let summary = "Annotate flash-style tile sizes on SDP attention (teaching)";
  let description = [{
    On ops already marked `aicom.scaled_dot_product_attention`, adds
    `aicom.flash_tile` plus `aicom.flash_tile_m/n` pseudo block sizes (no GPU
    kernel). Teaching companion to FlashAttention online softmax (Q14.2).
  }];
  let constructor = "mlir::aicom::createFlashAttentionTilePass()";
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

def GeluLegalize : Pass<"gelu-legalize", "func::FuncOp"> {
  let summary = "Canonicalize StableHLO GELU decomposition";
  let description = [{
    Recognizes exact GELU `0.5 * x * (1 + erf(x / sqrt(2)))` and annotates the
    final multiply with `aicom.gelu_canonicalized`.
  }];
  let constructor = "mlir::aicom::createGeluLegalizePass()";
  let dependentDialects = [
    "stablehlo::StablehloDialect",
    "chlo::ChloDialect",
  ];
}

def SwiGLULegalize : Pass<"swiglu-legalize", "func::FuncOp"> {
  let summary = "Canonicalize StableHLO SwiGLU decomposition";
  let description = [{
    Recognizes SwiGLU `silu(gate) * up` where `silu(x)=x/(1+exp(-x))` and
    annotates the final multiply with `aicom.swiglu_canonicalized`.
  }];
  let constructor = "mlir::aicom::createSwiGLULegalizePass()";
  let dependentDialects = [
    "stablehlo::StablehloDialect",
  ];
}

def QdqLegalize : Pass<"qdq-legalize", "func::FuncOp"> {
  let summary = "Canonicalize StableHLO Q/DQ MatMul decomposition";
  let description = [{
    Recognizes MatMul whose operands are ONNX-style dequantize chains
    `(x - zp) * scale` and annotates `aicom.qdq_matmul_canonicalized`.
  }];
  let constructor = "mlir::aicom::createQdqLegalizePass()";
  let dependentDialects = [
    "stablehlo::StablehloDialect",
  ];
}

def MatMulBiasFusion : Pass<"matmul-bias-fusion", "func::FuncOp"> {
  let summary = "Fuse StableHLO dot_general followed by constant bias add";
  let description = [{
    Matches `stablehlo.add` whose operand is `stablehlo.dot_general` and the
    other operand is a constant bias (optionally via `broadcast_in_dim`), then
    annotates the add with `aicom.matmul_bias_fused`.
  }];
  let constructor = "mlir::aicom::createMatMulBiasFusionPass()";
  let dependentDialects = [
    "stablehlo::StablehloDialect",
  ];
}

def HorizontalGemmFusion : Pass<"horizontal-gemm-fusion", "func::FuncOp"> {
  let summary = "Fuse shared-LHS dot_general outputs concatenated on last dim";
  let description = [{
    Matches `stablehlo.concatenate` whose operands are two
    `stablehlo.dot_general` ops sharing the same LHS (horizontal GEMM pattern:
    `concat(matmul(X,W1), matmul(X,W2))`), then annotates the concat with
    `aicom.horizontal_gemm_fused`.
  }];
  let constructor = "mlir::aicom::createHorizontalGemmFusionPass()";
  let dependentDialects = [
    "stablehlo::StablehloDialect",
  ];
}

def ElementwiseChainLegalize : Pass<"elementwise-chain-legalize", "func::FuncOp"> {
  let summary = "Annotate elementwise chain fused with ReLU boundary";
  let description = [{
    Recognizes ReLU written as `maximum(add/mul(...), zero)` or
    `clamp(0, add/mul(...), +inf)` and annotates the activation op with
    `aicom.elementwise_chain_fused`.
  }];
  let constructor = "mlir::aicom::createElementwiseChainLegalizePass()";
  let dependentDialects = [
    "stablehlo::StablehloDialect",
  ];
}

def ProducerConsumerLegalize : Pass<"producer-consumer-legalize", "func::FuncOp"> {
  let summary = "Annotate producer-consumer fusion (GEMM → softmax exp)";
  let description = [{
    Recognizes `stablehlo.exponential` whose operand is `subtract` from a
    `dot_general` score tensor (softmax decomposition) and annotates the exp
    with `aicom.producer_consumer_fused`.
  }];
  let constructor = "mlir::aicom::createProducerConsumerLegalizePass()";
  let dependentDialects = [
    "stablehlo::StablehloDialect",
  ];
}

def LayoutBridgeLegalize : Pass<"layout-bridge-legalize", "func::FuncOp"> {
  let summary = "Annotate NCHW conv + NHWC transpose layout fold boundary";
  let description = [{
    Teaching pass aligned with P5b: matches `stablehlo.transpose` with
  permutation `[0,2,3,1]` whose operand is `stablehlo.convolution` and
    annotates the transpose with `aicom.layout_folded` (annotation only).
  }];
  let constructor = "mlir::aicom::createLayoutBridgeLegalizePass()";
  let dependentDialects = [
    "stablehlo::StablehloDialect",
  ];
}

def KVCacheLegalize : Pass<"kvcache-legalize", "func::FuncOp"> {
  let summary = "Mark KV-cache role boundary on decode-step functions";
  let description = [{
    For `func.func` with `aicom.kv_role` of `"K"` or `"V"`, annotates
    `aicom.kvcache_boundary` (teaching placeholder, no runtime).
  }];
  let constructor = "mlir::aicom::createKVCacheLegalizePass()";
  let dependentDialects = [
    "func::FuncDialect",
  ];
}

def LayoutFold : Pass<"layout-fold", "func::FuncOp"> {
  let summary = "Fold NCHW conv + NHWC transpose into one NHWC-output conv";
  let description = [{
    Teaching pass (A2): when a `stablehlo.transpose` with perm [0,2,3,1]
    follows a `stablehlo.convolution` with constant weights, rewrites to a
    single convolution with NHWC output dim_numbers and marks
    `aicom.layout_folded`. Eliminates the transpose op from the graph.
  }];
  let constructor = "mlir::aicom::createLayoutFoldPass()";
  let dependentDialects = [
    "stablehlo::StablehloDialect",
  ];
}

def BroadcastSimplify : Pass<"broadcast-simplify", "func::FuncOp"> {
  let summary = "Fold redundant stablehlo.broadcast_in_dim chains";
  let description = [{
    Teaching graph-opt pass aligned with P4 tier 2 numpy-style broadcast:
    - removes identity broadcasts (operand shape equals result shape)
    - fuses nested `broadcast_in_dim(broadcast_in_dim(x))` into one broadcast
    Annotates fused broadcasts with `aicom.broadcast_simplified`.
  }];
  let constructor = "mlir::aicom::createBroadcastSimplifyPass()";
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

def CustomLinalgTile : Pass<"custom-linalg-tile", "func::FuncOp"> {
  let summary = "Tile static linalg.matmul with 2x2x2 tiles (teaching)";
  let description = [{
    Strip-mines `linalg.matmul` on tensor operands using MLIR
    `tileLinalgOp` and annotates the tiled op with `aicom.linalg_tiled`.
  }];
  let constructor = "mlir::aicom::createCustomLinalgTilePass()";
  let dependentDialects = [
    "linalg::LinalgDialect",
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
