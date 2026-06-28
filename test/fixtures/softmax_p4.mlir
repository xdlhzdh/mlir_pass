module {
  func.func @main(%arg0: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %0 = stablehlo.constant dense<-3.402823e+38> : tensor<f32>
    %1 = stablehlo.constant dense<0.0> : tensor<f32>
    %2 = stablehlo.reduce(%arg0 init: %0) applies stablehlo.maximum across dimensions = [1] : (tensor<2x4xf32>, tensor<f32>) -> tensor<2xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xf32>) -> tensor<2x4xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<2x4xf32>
    %5 = stablehlo.exponential %4 : tensor<2x4xf32>
    %6 = stablehlo.reduce(%5 init: %1) applies stablehlo.add across dimensions = [1] : (tensor<2x4xf32>, tensor<f32>) -> tensor<2xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0] : (tensor<2xf32>) -> tensor<2x4xf32>
    %8 = stablehlo.divide %5, %7 : tensor<2x4xf32>
    return %8 : tensor<2x4xf32>
  }
}
