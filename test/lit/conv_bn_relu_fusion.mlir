// RUN: %pipe-demo --input=%s --pipeline-stop-after=fusion 2>&1 | FileCheck %s
// CHECK-NOT: batch_norm_inference
// CHECK-NOT: stablehlo.maximum
// CHECK: stablehlo.convolution
// CHECK: stablehlo.add
// CHECK: stablehlo.clamp

module {
  func.func @inference() -> tensor<1x2x2x2xf32> {
    %input = stablehlo.constant dense<[[[[1.0], [2.0]], [[3.0], [4.0]]]]> : tensor<1x2x2x1xf32>
    %weight = stablehlo.constant dense<[[[[0.5, 0.5]]]]> : tensor<1x1x1x2xf32>
    %scale = stablehlo.constant dense<1.0> : tensor<2xf32>
    %offset = stablehlo.constant dense<0.0> : tensor<2xf32>
    %mean = stablehlo.constant dense<0.0> : tensor<2xf32>
    %var = stablehlo.constant dense<1.0> : tensor<2xf32>
    %zero = stablehlo.constant dense<0.0> : tensor<1x2x2x2xf32>

    %conv = stablehlo.convolution(%input, %weight)
        dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
        window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1],
                  rhs_dilate = [1, 1], reverse = [0, 0]}
        {batch_group_count = 1 : i64, feature_group_count = 1 : i64,
         precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]}
        : (tensor<1x2x2x1xf32>, tensor<1x1x1x2xf32>) -> tensor<1x2x2x2xf32>

    %bn = "stablehlo.batch_norm_inference"(%conv, %scale, %offset, %mean, %var) {
      epsilon = 1.000000e-05 : f32,
      feature_index = 3 : i64
    } : (tensor<1x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> tensor<1x2x2x2xf32>

    %relu = stablehlo.maximum %bn, %zero : tensor<1x2x2x2xf32>
    return %relu : tensor<1x2x2x2xf32>
  }
}
