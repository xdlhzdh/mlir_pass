// Combined payload + Buddy-style tiling transform for a 256³ f16→f32 GEMM.
// Run: mlir-opt matmul_tiling.mlir --transform-interpreter \
//        --pass-pipeline='builtin.module(transform-interpreter{disable-expensive-checks=true})'
module attributes {transform.with_named_sequence} {
  func.func @matmul(%A: tensor<256x256xf16>,
                    %B: tensor<256x256xf16>,
                    %C: tensor<256x256xf32>) -> tensor<256x256xf32> {
    %0 = linalg.matmul
           ins(%A, %B : tensor<256x256xf16>, tensor<256x256xf16>)
           outs(%C : tensor<256x256xf32>) -> tensor<256x256xf32>
    return %0 : tensor<256x256xf32>
  }

  transform.named_sequence @__transform_main(
      %root: !transform.any_op) {
    %matmul =
      transform.structured.match ops{["linalg.matmul"]}
      in %root
      : (!transform.any_op) -> !transform.any_op

    %cta_matmul, %cta_forall =
      transform.structured.tile_using_forall %matmul
        tile_sizes [128, 128]
        (mapping = [#gpu.block<y>, #gpu.block<x>])
      : (!transform.any_op)
        -> (!transform.any_op, !transform.any_op)

    %k_tiled, %k_loop =
      transform.structured.tile_using_for %cta_matmul
        tile_sizes [0, 0, 64]
      : (!transform.any_op)
        -> (!transform.any_op, !transform.any_op)

    %warp_matmul, %warp_forall =
      transform.structured.tile_using_forall %k_tiled
        num_threads [4, 4]
        (mapping = [#gpu.warp<y>, #gpu.warp<x>])
      : (!transform.any_op)
        -> (!transform.any_op, !transform.any_op)

    %func =
      transform.structured.match ops{["func.func"]}
      in %root
      : (!transform.any_op) -> !transform.any_op

    transform.apply_patterns to %func {
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.scf.for_loop_canonicalization
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func : !transform.any_op

    %func_v =
      transform.structured.vectorize_children_and_apply_patterns %func
        {fold_type_extensions_into_contract}
      : (!transform.any_op) -> !transform.any_op

    %func_e =
      transform.apply_registered_pass "empty-tensor-to-alloc-tensor"
        to %func_v
      : (!transform.any_op) -> !transform.any_op

    %root_b =
      transform.bufferization.one_shot_bufferize %root
        {bufferize_function_boundaries = true}
      : (!transform.any_op) -> !transform.any_op

    %func2 =
      transform.structured.match ops{["func.func"]}
      in %root_b
      : (!transform.any_op) -> !transform.any_op

    %gpu_launch =
      transform.gpu.map_forall_to_blocks %func2
        {generate_gpu_launch}
      : (!transform.any_op) -> !transform.any_op

    // Non-linear #gpu.warp{y,x} with 4×4 warps:
    //   X needs 4*32 threads, Y needs 4 warps → [128, 4, 1]
    %gpu_launch2 =
      transform.gpu.map_nested_forall_to_threads %gpu_launch
        block_dims = [128, 4, 1]
        warp_size = 32
      : (!transform.any_op) -> !transform.any_op

    %func3 =
      transform.structured.match ops{["func.func"]}
      in %root_b
      : (!transform.any_op) -> !transform.any_op

    transform.apply_patterns to %func3 {
      transform.apply_patterns.gpu.unroll_vectors_subgroup_mma [16, 16, 16]
    } : !transform.any_op

    %func4 =
      transform.apply_registered_pass "convert-vector-to-gpu"
        with options = {"use-nvgpu" = true}
        to %func3
      : (!transform.any_op) -> !transform.any_op

    transform.apply_patterns to %func4 {
      transform.apply_patterns.canonicalization
      transform.apply_patterns.vector.lower_masks
    } : !transform.any_op
    transform.apply_cse to %func4 : !transform.any_op

    transform.yield
  }
}
