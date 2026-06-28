module {
  func.func @main(%arg0: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %0 = stablehlo.constant dense<[1.0, 1.0, 1.0, 1.0]> : tensor<4xf32>
    %1 = stablehlo.constant dense<2.0> : tensor<f32>
    %2 = stablehlo.constant dense<1.0e-05> : tensor<f32>
    %3 = stablehlo.multiply %arg0, %arg0 : tensor<2x4xf32>
    %4 = stablehlo.constant dense<0.0> : tensor<f32>
    %5 = stablehlo.reduce(%3 init: %4) applies stablehlo.add across dimensions = [1] : (tensor<2x4xf32>, tensor<f32>) -> tensor<2xf32>
    %6 = stablehlo.reshape %5 : (tensor<2xf32>) -> tensor<2x1xf32>
    %7 = stablehlo.constant dense<4.0> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
    %9 = stablehlo.divide %6, %8 : tensor<2x1xf32>
    %10 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
    %11 = stablehlo.add %9, %10 : tensor<2x1xf32>
    %12 = stablehlo.sqrt %11 : tensor<2x1xf32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x4xf32>
    %14 = stablehlo.divide %arg0, %13 : tensor<2x4xf32>
    %15 = stablehlo.broadcast_in_dim %0, dims = [1] : (tensor<4xf32>) -> tensor<2x4xf32>
    %16 = stablehlo.multiply %14, %15 : tensor<2x4xf32>
    return %16 : tensor<2x4xf32>
  }
}
