module {
  func.func @inference(%arg0: tensor<2x3xf32>) -> tensor<2x4xf32> {
    %0 = stablehlo.constant dense<[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]> : tensor<3x2xf32>
    %1 = stablehlo.constant dense<[[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]]> : tensor<3x2xf32>
    %2 = stablehlo.dot_general %arg0, %0, batching_dims = [] x [], contracting_dims = [1] x [0] : (tensor<2x3xf32>, tensor<3x2xf32>) -> tensor<2x2xf32>
    %3 = stablehlo.dot_general %arg0, %1, batching_dims = [] x [], contracting_dims = [1] x [0] : (tensor<2x3xf32>, tensor<3x2xf32>) -> tensor<2x2xf32>
    %4 = stablehlo.concatenate %2, %3, dim = 1 : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x4xf32>
    return %4 : tensor<2x4xf32>
  }
}
