module {
  func.func @inference(%arg0: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %0 = stablehlo.constant dense<[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]> : tensor<4x4xf32>
    %1 = stablehlo.dot_general %arg0, %0, batching_dims = [] x [], contracting_dims = [1] x [0] : (tensor<2x4xf32>, tensor<4x4xf32>) -> tensor<2x4xf32>
    %2 = stablehlo.constant dense<-3.402823e+38> : tensor<f32>
    %3 = stablehlo.constant dense<0.0> : tensor<f32>
    %4 = stablehlo.reduce(%1 init: %2) applies stablehlo.maximum across dimensions = [1] : (tensor<2x4xf32>, tensor<f32>) -> tensor<2xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0] : (tensor<2xf32>) -> tensor<2x4xf32>
    %6 = stablehlo.subtract %1, %5 : tensor<2x4xf32>
    %7 = stablehlo.exponential %6 : tensor<2x4xf32>
    %8 = stablehlo.reduce(%7 init: %3) applies stablehlo.add across dimensions = [1] : (tensor<2x4xf32>, tensor<f32>) -> tensor<2xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0] : (tensor<2xf32>) -> tensor<2x4xf32>
    %10 = stablehlo.divide %7, %9 : tensor<2x4xf32>
    return %10 : tensor<2x4xf32>
  }
}
