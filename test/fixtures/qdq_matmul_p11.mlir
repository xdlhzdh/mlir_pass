module {
  func.func @main(%arg0: tensor<2x3xf32>) -> tensor<2x4xf32> {
    %0 = stablehlo.constant dense<[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]> : tensor<3x4xf32>
    %1 = stablehlo.constant dense<0.1> : tensor<f32>
    %2 = stablehlo.constant dense<0> : tensor<i8>
    %3 = stablehlo.constant dense<0.1> : tensor<f32>
    %4 = stablehlo.constant dense<0> : tensor<i8>
    %5 = stablehlo.constant dense<0.2> : tensor<f32>
    %6 = stablehlo.constant dense<0> : tensor<i8>
    %7 = stablehlo.convert %2 : (tensor<i8>) -> tensor<f32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [] : (tensor<f32>) -> tensor<2x3xf32>
    %9 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<f32>) -> tensor<2x3xf32>
    %10 = stablehlo.subtract %arg0, %8 : tensor<2x3xf32>
    %11 = stablehlo.multiply %10, %9 : tensor<2x3xf32>
    %12 = stablehlo.convert %4 : (tensor<i8>) -> tensor<f32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [] : (tensor<f32>) -> tensor<3x4xf32>
    %14 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<3x4xf32>
    %15 = stablehlo.subtract %0, %13 : tensor<3x4xf32>
    %16 = stablehlo.multiply %15, %14 : tensor<3x4xf32>
    %17 = stablehlo.dot_general %11, %16, batching_dims = [] x [], contracting_dims = [1] x [0] : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
    %18 = stablehlo.convert %6 : (tensor<i8>) -> tensor<f32>
    %19 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<2x4xf32>
    %20 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<f32>) -> tensor<2x4xf32>
    %21 = stablehlo.subtract %17, %19 : tensor<2x4xf32>
    %22 = stablehlo.multiply %21, %20 : tensor<2x4xf32>
    return %22 : tensor<2x4xf32>
  }
}
