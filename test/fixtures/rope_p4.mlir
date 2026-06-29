module {
  func.func @inference(%arg0: tensor<1x2x4xf32>) -> tensor<1x2x4xf32> {
    %0 = stablehlo.constant dense<[[[1.0, 1.0, 1.0, 1.0], [0.540302, 0.540302, 0.99995, 0.99995]]]> : tensor<1x2x4xf32>
    %1 = stablehlo.constant dense<[[[0.0, 0.0, 0.0, 0.0], [0.841471, 0.841471, 0.00999983, 0.00999983]]]> : tensor<1x2x4xf32>
    %2 = stablehlo.constant dense<-1.0> : tensor<f32>
    %3 = stablehlo.constant dense<[0, 0, 0]> : tensor<3xi64>
    %4 = stablehlo.constant dense<[1, 2, 2]> : tensor<3xi64>
    %5 = stablehlo.constant dense<[0, 0, 2]> : tensor<3xi64>
    %6 = stablehlo.constant dense<[1, 2, 4]> : tensor<3xi64>
    %7 = stablehlo.constant dense<[0, 1, 2]> : tensor<3xi64>
    %8 = stablehlo.slice %arg0 [0:1, 0:2, 0:2] : (tensor<1x2x4xf32>) -> tensor<1x2x2xf32>
    %9 = stablehlo.slice %arg0 [0:1, 0:2, 2:4] : (tensor<1x2x4xf32>) -> tensor<1x2x2xf32>
    %10 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<1x2x2xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<1x2x2xf32>
    %12 = stablehlo.concatenate %11, %8, dim = 2 : (tensor<1x2x2xf32>, tensor<1x2x2xf32>) -> tensor<1x2x4xf32>
    %13 = stablehlo.multiply %arg0, %0 : tensor<1x2x4xf32>
    %14 = stablehlo.multiply %12, %1 : tensor<1x2x4xf32>
    %15 = stablehlo.add %13, %14 : tensor<1x2x4xf32>
    return %15 : tensor<1x2x4xf32>
  }
}
