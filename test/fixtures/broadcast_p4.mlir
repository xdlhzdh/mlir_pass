module {
  func.func @inference(%arg0: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
    %0 = stablehlo.constant dense<[1.0, 1.0, 1.0, 1.0]> : tensor<4xf32>
    %1 = stablehlo.constant dense<[[[2.0], [2.0], [2.0]]]> : tensor<1x3x1xf32>
    %2 = stablehlo.broadcast_in_dim %0, dims = [2] : (tensor<4xf32>) -> tensor<2x3x4xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<2x3x4xf32>
    %4 = stablehlo.broadcast_in_dim %1, dims = [0, 1, 2] : (tensor<1x3x1xf32>) -> tensor<2x3x4xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<2x3x4xf32>
    return %5 : tensor<2x3x4xf32>
  }
}
