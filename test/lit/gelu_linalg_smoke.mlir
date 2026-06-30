// RUN: %pipe-demo --input=%s --pipeline-stop-after=linalg 2>&1 | FileCheck %s

// After fusion + chlo-legalize-to-stablehlo + stablehlo-legalize-to-linalg:
// CHECK-NOT: chlo.erf
// CHECK: linalg.

module {
  func.func @inference(%arg0: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %half = stablehlo.constant dense<0.5> : tensor<f32>
    %one = stablehlo.constant dense<1.0> : tensor<f32>
    %sqrt2 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %half_b = stablehlo.broadcast_in_dim %half, dims = [] : (tensor<f32>) -> tensor<2x4xf32>
    %scaled = stablehlo.divide %arg0, %half_b : tensor<2x4xf32>
    %sqrt2_b = stablehlo.broadcast_in_dim %sqrt2, dims = [] : (tensor<f32>) -> tensor<2x4xf32>
    %div = stablehlo.divide %scaled, %sqrt2_b : tensor<2x4xf32>
    %erf = chlo.erf %div : tensor<2x4xf32> -> tensor<2x4xf32>
    %one_b = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<2x4xf32>
    %inner = stablehlo.add %erf, %one_b : tensor<2x4xf32>
    %res = stablehlo.multiply %half_b, %inner {aicom.gelu_canonicalized} : tensor<2x4xf32>
    return %res : tensor<2x4xf32>
  }
}
