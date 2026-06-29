// RUN: %pipe-demo --input=%s --pipeline-stop-after=fusion 2>&1 | FileCheck %s
// CHECK: func.func @decode_step
// CHECK-SAME: aicom.kvcache_boundary

module {
  func.func @decode_step() -> tensor<1x2x4xf32> attributes { aicom.kv_role = "K" } {
    %x = stablehlo.constant dense<[[[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]]]> : tensor<1x2x4xf32>
    %cache_slot = stablehlo.constant dense<0.0> : tensor<1x2x4xf32>
    %out = stablehlo.add %x, %cache_slot : tensor<1x2x4xf32>
    return %out : tensor<1x2x4xf32>
  }
}
