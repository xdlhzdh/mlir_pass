module {
  func.func @main(%arg0: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %0 = stablehlo.constant dense<[1.0, 1.0, 1.0, 1.0]> : tensor<4xf32>
    %1 = stablehlo.constant dense<[0.0, 0.0, 0.0, 0.0]> : tensor<4xf32>
    %2 = stablehlo.constant dense<1.0e-05> : tensor<f32>
    %3 = stablehlo.constant dense<0.0> : tensor<f32>
    %4 = stablehlo.reduce(%arg0 init: %3) applies stablehlo.add across dimensions = [1] : (tensor<2x4xf32>, tensor<f32>) -> tensor<2xf32>
    %5 = stablehlo.reshape %4 : (tensor<2xf32>) -> tensor<2x1xf32>
    %6 = stablehlo.constant dense<4.0> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
    %8 = stablehlo.divide %5, %7 : tensor<2x1xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x4xf32>
    %10 = stablehlo.subtract %arg0, %9 : tensor<2x4xf32>
    %11 = stablehlo.multiply %10, %10 : tensor<2x4xf32>
    %12 = stablehlo.constant dense<0.0> : tensor<f32>
    %13 = stablehlo.reduce(%11 init: %12) applies stablehlo.add across dimensions = [1] : (tensor<2x4xf32>, tensor<f32>) -> tensor<2xf32>
    %14 = stablehlo.reshape %13 : (tensor<2xf32>) -> tensor<2x1xf32>
    %15 = stablehlo.constant dense<4.0> : tensor<f32>
    %16 = stablehlo.broadcast_in_dim %15, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
    %17 = stablehlo.divide %14, %16 : tensor<2x1xf32>
    %18 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
    %19 = stablehlo.add %17, %18 : tensor<2x1xf32>
    %20 = stablehlo.sqrt %19 : tensor<2x1xf32>
    %21 = stablehlo.broadcast_in_dim %20, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x4xf32>
    %22 = stablehlo.divide %10, %21 : tensor<2x4xf32>
    %23 = stablehlo.broadcast_in_dim %0, dims = [1] : (tensor<4xf32>) -> tensor<2x4xf32>
    %24 = stablehlo.multiply %22, %23 : tensor<2x4xf32>
    %25 = stablehlo.broadcast_in_dim %1, dims = [1] : (tensor<4xf32>) -> tensor<2x4xf32>
    %26 = stablehlo.add %24, %25 : tensor<2x4xf32>
    return %26 : tensor<2x4xf32>
  }
}
