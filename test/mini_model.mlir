// Full mini inference graph: Conv + BN + ReLU + MatMul + Add (nullary, small tensors).
module {
  func.func @inference() -> tensor<1x2xf32> {
    %input = stablehlo.constant dense<[[[[1.0], [2.0]], [[3.0], [4.0]]]]> : tensor<1x2x2x1xf32>
    %conv_w = stablehlo.constant dense<[[[[0.5, 0.5]]]]> : tensor<1x1x1x2xf32>
    %bn_scale = stablehlo.constant dense<1.0> : tensor<2xf32>
    %bn_offset = stablehlo.constant dense<0.0> : tensor<2xf32>
    %bn_mean = stablehlo.constant dense<0.0> : tensor<2xf32>
    %bn_var = stablehlo.constant dense<1.0> : tensor<2xf32>
    %zero = stablehlo.constant dense<0.0> : tensor<1x2x2x2xf32>

    %conv = stablehlo.convolution(%input, %conv_w)
        dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
        window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1],
                  rhs_dilate = [1, 1], reverse = [0, 0]}
        {batch_group_count = 1 : i64, feature_group_count = 1 : i64,
         precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]}
        : (tensor<1x2x2x1xf32>, tensor<1x1x1x2xf32>) -> tensor<1x2x2x2xf32>

    %bn = "stablehlo.batch_norm_inference"(%conv, %bn_scale, %bn_offset, %bn_mean, %bn_var) {
      epsilon = 1.000000e-05 : f32,
      feature_index = 3 : i64
    } : (tensor<1x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> tensor<1x2x2x2xf32>

    %relu = stablehlo.maximum %bn, %zero : tensor<1x2x2x2xf32>

    %flat = stablehlo.reshape %relu : (tensor<1x2x2x2xf32>) -> tensor<1x8xf32>
    %fc_w = stablehlo.constant dense<[
      [1.0, 0.0],
      [0.0, 1.0],
      [0.0, 0.0],
      [0.0, 0.0],
      [0.0, 0.0],
      [0.0, 0.0],
      [0.0, 0.0],
      [0.0, 0.0]
    ]> : tensor<8x2xf32>
    %mm = stablehlo.dot_general %flat, %fc_w,
      contracting_dims = [1] x [0],
      precision = [DEFAULT, DEFAULT]
      : (tensor<1x8xf32>, tensor<8x2xf32>) -> tensor<1x2xf32>
    %fc_b = stablehlo.constant dense<0.1> : tensor<2xf32>
    %fc_bcast = stablehlo.broadcast_in_dim %fc_b, dims = [1]
        : (tensor<2xf32>) -> tensor<1x2xf32>
    %out = stablehlo.add %mm, %fc_bcast : tensor<1x2xf32>
    return %out : tensor<1x2xf32>
  }
}
