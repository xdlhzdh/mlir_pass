// RUN: %pipe-demo --input=%s --pipeline-stop-after=fusion 2>&1 | FileCheck %s
// NCHW activations with an OIHW kernel: the BN feature_index (1) is not the
// kernel output-feature dimension (0). Cin == Cout so a wrong axis would still
// type-check; the folded weight must be scaled per output channel.
// CHECK-NOT: batch_norm_inference
// CHECK-DAG: stablehlo.constant dense<{{\[\[\[\[}}2.000000e+00]], {{\[\[}}4.000000e+00]]], {{\[\[\[}}9.000000e+00]], {{\[\[}}1.200000e+01]]]]> : tensor<2x2x1x1xf32>
// CHECK-DAG: stablehlo.constant dense<[0.000000e+00, -5.000000e-01]> : tensor<2xf32>
// CHECK: stablehlo.convolution
// CHECK-SAME: [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1]
// CHECK: stablehlo.broadcast_in_dim {{.*}} dims = [1]
// CHECK: stablehlo.add

module {
  func.func @inference(%input: tensor<1x2x3x3xf32>) -> tensor<1x2x3x3xf32> {
    %weight = stablehlo.constant dense<[[[[1.0]], [[2.0]]], [[[3.0]], [[4.0]]]]> : tensor<2x2x1x1xf32>
    %scale = stablehlo.constant dense<[2.0, 3.0]> : tensor<2xf32>
    %offset = stablehlo.constant dense<[0.5, 1.0]> : tensor<2xf32>
    %mean = stablehlo.constant dense<[0.25, 0.5]> : tensor<2xf32>
    %var = stablehlo.constant dense<1.0> : tensor<2xf32>
    %conv = stablehlo.convolution(%input, %weight)
        dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
        window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1],
                  rhs_dilate = [1, 1], reverse = [0, 0]}
        {batch_group_count = 1 : i64, feature_group_count = 1 : i64,
         precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]}
        : (tensor<1x2x3x3xf32>, tensor<2x2x1x1xf32>) -> tensor<1x2x3x3xf32>
    %bn = "stablehlo.batch_norm_inference"(%conv, %scale, %offset, %mean, %var) {
      epsilon = 0.000000e+00 : f32, feature_index = 1 : i64
    } : (tensor<1x2x3x3xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> tensor<1x2x3x3xf32>
    return %bn : tensor<1x2x3x3xf32>
  }
}
