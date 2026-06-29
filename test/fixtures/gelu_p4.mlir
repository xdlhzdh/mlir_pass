module {
  func.func @inference(%arg0: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %0 = stablehlo.constant dense<0.5> : tensor<f32>
    %1 = stablehlo.constant dense<1.0> : tensor<f32>
    %2 = stablehlo.constant dense<2.0> : tensor<f32>
    %3 = stablehlo.sqrt %2 : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<2x4xf32>
    %5 = stablehlo.divide %arg0, %4 : tensor<2x4xf32>
    %6 = chlo.erf %5 : tensor<2x4xf32> -> tensor<2x4xf32>
    %7 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<f32>) -> tensor<2x4xf32>
    %8 = stablehlo.add %7, %6 : tensor<2x4xf32>
    %9 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<2x4xf32>
    %10 = stablehlo.multiply %arg0, %9 : tensor<2x4xf32>
    %11 = stablehlo.multiply %10, %8 : tensor<2x4xf32>
    return %11 : tensor<2x4xf32>
  }
}
