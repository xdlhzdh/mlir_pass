// Matmul payload sized for a single warp Tensor Core MMA.sync.
// Tiling / vectorize / gpu mapping are NOT in the transform below:
// they are pre-baked into this kernel:
//   - 1 CTA × 32 threads  =  one warp (gpu.launch already present)
//   - A: 16x16xf16, B: 16x8xf16, C: 16x8xf16  =  one m16n8k16 instruction
// rewrite_matmul_as_mma_sync then distributes fragment loads/stores by laneId
// and emits nvgpu.mma.sync. For CTA/K/warp tiling see matmul_tiling.mlir.
module attributes {transform.with_named_sequence} {
  func.func @matmul_kernel(%A: memref<16x16xf16>,
                           %B: memref<16x8xf16>,
                           %C: memref<16x8xf16>) {
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
               threads(%tx, %ty, %tz) in (%bdx = %c32, %bdy = %c1, %bdz = %c1) {
      linalg.matmul
          ins(%A, %B : memref<16x16xf16>, memref<16x8xf16>)
          outs(%C : memref<16x8xf16>)
      gpu.terminator
    }
    return
  }

  transform.named_sequence @__transform_main(
      %root: !transform.any_op) {
    %matmul =
      transform.structured.match ops{["linalg.matmul"]}
      in %root
      : (!transform.any_op) -> !transform.any_op

    // Tensorize: linalg.matmul → nvgpu.mma.sync (+ fragment loads/stores).
    transform.nvgpu.rewrite_matmul_as_mma_sync %matmul
      : (!transform.any_op) -> ()

    transform.yield
  }
}
