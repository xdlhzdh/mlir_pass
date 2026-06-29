module {
  func.func @inference(%arg0: tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32> {
    %0 = stablehlo.constant dense<[[[[0.5]], [[0.5]]], [[[0.5]], [[0.5]]]]> : tensor<2x2x1x1xf32>
    %1 = stablehlo.convolution(%arg0, %0)
          dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
          window = {stride = [1, 1], pad = [[0, 0], [0, 0]]}
          {feature_group_count = 1 : i64, batch_group_count = 1 : i64}
          : (tensor<1x2x2x2xf32>, tensor<2x2x1x1xf32>) -> tensor<1x2x2x2xf32>
    %2 = stablehlo.transpose %1, dims = [0, 2, 3, 1] : (tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
    return %2 : tensor<1x2x2x2xf32>
  }
}
