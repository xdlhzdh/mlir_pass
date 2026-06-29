module {
  func.func @main(%arg0: tensor<2x3xf32>) -> tensor<2x4xf32> {
    %0 = stablehlo.constant dense<[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]> : tensor<3x4xf32>
    %1 = stablehlo.constant dense<0.1> : tensor<f32>
    %2 = stablehlo.constant dense<0.0> : tensor<f32>
    %3 = stablehlo.constant dense<0.2> : tensor<f32>
    %4 = stablehlo.constant dense<0.0> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<2x3xf32>
    %6 = stablehlo.subtract %arg0, %5 : tensor<2x3xf32>
    %7 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<f32>) -> tensor<2x3xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<2x3xf32>
    %9 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<f32>) -> tensor<3x4xf32>
    %10 = stablehlo.subtract %0, %9 : tensor<3x4xf32>
    %11 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<3x4xf32>
    %12 = stablehlo.multiply %10, %11 : tensor<3x4xf32>
    %13 = stablehlo.dot_general %8, %12, batching_dims = [] x [], contracting_dims = [1] x [0] : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
    return %13 : tensor<2x4xf32>
  }
}
