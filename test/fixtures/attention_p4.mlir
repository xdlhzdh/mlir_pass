module {
  func.func @main(%arg0: tensor<1x2x4xf32>, %arg1: tensor<1x4x2xf32>, %arg2: tensor<1x2x4xf32>) -> tensor<1x2x4xf32> {
    %0 = stablehlo.constant dense<0.5> : tensor<f32>
    %1 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<1x2x4xf32>, tensor<1x4x2xf32>) -> tensor<1x2x2xf32>
    %2 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<1x2x2xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<1x2x2xf32>
    %4 = stablehlo.constant dense<-3.402823e+38> : tensor<f32>
    %5 = stablehlo.constant dense<0.0> : tensor<f32>
    %6 = stablehlo.reduce(%3 init: %4) applies stablehlo.maximum across dimensions = [2] : (tensor<1x2x2xf32>, tensor<f32>) -> tensor<1x2xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1] : (tensor<1x2xf32>) -> tensor<1x2x2xf32>
    %8 = stablehlo.subtract %3, %7 : tensor<1x2x2xf32>
    %9 = stablehlo.exponential %8 : tensor<1x2x2xf32>
    %10 = stablehlo.reduce(%9 init: %5) applies stablehlo.add across dimensions = [2] : (tensor<1x2x2xf32>, tensor<f32>) -> tensor<1x2xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<1x2xf32>) -> tensor<1x2x2xf32>
    %12 = stablehlo.divide %9, %11 : tensor<1x2x2xf32>
    %13 = stablehlo.dot_general %12, %arg2, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<1x2x2xf32>, tensor<1x2x4xf32>) -> tensor<1x2x4xf32>
    return %13 : tensor<1x2x4xf32>
  }
}
