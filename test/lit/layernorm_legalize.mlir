// RUN: %pipe-demo --input=%s --pipeline-stop-after=fusion 2>&1 | FileCheck %s
// CHECK: stablehlo.subtract
// CHECK: stablehlo.divide
// CHECK: aicom.layernorm_canonicalized

module {
  func.func @inference(%arg0: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %gamma = stablehlo.constant dense<[1.0, 1.0, 1.0, 1.0]> : tensor<4xf32>
    %beta = stablehlo.constant dense<[0.0, 0.0, 0.0, 0.0]> : tensor<4xf32>
    %eps = stablehlo.constant dense<1.0e-05> : tensor<f32>
    %zero = stablehlo.constant dense<0.0> : tensor<f32>
    %four = stablehlo.constant dense<4.0> : tensor<f32>
    %mean = stablehlo.reduce(%arg0 init: %zero) applies stablehlo.add across dimensions = [1]
        : (tensor<2x4xf32>, tensor<f32>) -> tensor<2xf32>
    %mean_r = stablehlo.reshape %mean : (tensor<2xf32>) -> tensor<2x1xf32>
    %four_b = stablehlo.broadcast_in_dim %four, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
    %mean_v = stablehlo.divide %mean_r, %four_b : tensor<2x1xf32>
    %mean_bc = stablehlo.broadcast_in_dim %mean_v, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x4xf32>
    %centered = stablehlo.subtract %arg0, %mean_bc : tensor<2x4xf32>
    %sq = stablehlo.multiply %centered, %centered : tensor<2x4xf32>
    %var = stablehlo.reduce(%sq init: %zero) applies stablehlo.add across dimensions = [1]
        : (tensor<2x4xf32>, tensor<f32>) -> tensor<2xf32>
    %var_r = stablehlo.reshape %var : (tensor<2xf32>) -> tensor<2x1xf32>
    %var_m = stablehlo.divide %var_r, %four_b : tensor<2x1xf32>
    %eps_b = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
    %var_eps = stablehlo.add %var_m, %eps_b : tensor<2x1xf32>
    %denom = stablehlo.sqrt %var_eps : tensor<2x1xf32>
    %denom_bc = stablehlo.broadcast_in_dim %denom, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x4xf32>
    %norm = stablehlo.divide %centered, %denom_bc : tensor<2x4xf32>
    %gamma_b = stablehlo.broadcast_in_dim %gamma, dims = [1] : (tensor<4xf32>) -> tensor<2x4xf32>
    %scaled = stablehlo.multiply %norm, %gamma_b : tensor<2x4xf32>
    %beta_b = stablehlo.broadcast_in_dim %beta, dims = [1] : (tensor<4xf32>) -> tensor<2x4xf32>
    %out = stablehlo.add %scaled, %beta_b : tensor<2x4xf32>
    return %out : tensor<2x4xf32>
  }
}
