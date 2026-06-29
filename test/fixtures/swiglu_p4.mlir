module {
  func.func @inference(%arg0: tensor<2x4xf32>, %arg1: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %0 = stablehlo.constant dense<1.0> : tensor<f32>
    %1 = stablehlo.negate %arg0 : tensor<2x4xf32>
    %2 = stablehlo.exponential %1 : tensor<2x4xf32>
    %3 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<2x4xf32>
    %4 = stablehlo.add %3, %2 : tensor<2x4xf32>
    %5 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<2x4xf32>
    %6 = stablehlo.divide %5, %4 : tensor<2x4xf32>
    %7 = stablehlo.multiply %arg0, %6 : tensor<2x4xf32>
    %8 = stablehlo.multiply %7, %arg1 : tensor<2x4xf32>
    return %8 : tensor<2x4xf32>
  }
}
