module {
  func.func @inference(%arg0: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %0 = stablehlo.constant dense<[1.0, 1.0, 1.0, 1.0]> : tensor<4xf32>
    %1 = stablehlo.constant dense<2.0> : tensor<f32>
    %2 = stablehlo.constant dense<1.0e-05> : tensor<f32>
    %3 = stablehlo.constant dense<[[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]]> : tensor<1x4x2xf32>
    %4 = stablehlo.constant dense<[[[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]]]> : tensor<1x2x4xf32>
    %5 = stablehlo.constant dense<0.5> : tensor<f32>
    %6 = stablehlo.constant dense<[1, 2, 4]> : tensor<3xi64>
    %7 = stablehlo.constant dense<[2, 4]> : tensor<2xi64>
    %8 = stablehlo.constant dense<1.0> : tensor<f32>
    %9 = stablehlo.multiply %arg0, %arg0 : tensor<2x4xf32>
    %10 = stablehlo.constant dense<0.0> : tensor<f32>
    %11 = stablehlo.reduce(%9 init: %10) applies stablehlo.add across dimensions = [1] : (tensor<2x4xf32>, tensor<f32>) -> tensor<2xf32>
    %12 = stablehlo.reshape %11 : (tensor<2xf32>) -> tensor<2x1xf32>
    %13 = stablehlo.constant dense<4.0> : tensor<f32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
    %15 = stablehlo.divide %12, %14 : tensor<2x1xf32>
    %16 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
    %17 = stablehlo.add %15, %16 : tensor<2x1xf32>
    %18 = stablehlo.sqrt %17 : tensor<2x1xf32>
    %19 = stablehlo.broadcast_in_dim %18, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x4xf32>
    %20 = stablehlo.divide %arg0, %19 : tensor<2x4xf32>
    %21 = stablehlo.broadcast_in_dim %0, dims = [1] : (tensor<4xf32>) -> tensor<2x4xf32>
    %22 = stablehlo.multiply %20, %21 : tensor<2x4xf32>
    %23 = stablehlo.reshape %22 : (tensor<2x4xf32>) -> tensor<1x2x4xf32>
    %24 = stablehlo.dot_general %23, %3, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<1x2x4xf32>, tensor<1x4x2xf32>) -> tensor<1x2x2xf32>
    %25 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<f32>) -> tensor<1x2x2xf32>
    %26 = stablehlo.multiply %24, %25 : tensor<1x2x2xf32>
    %27 = stablehlo.constant dense<-3.402823e+38> : tensor<f32>
    %28 = stablehlo.constant dense<0.0> : tensor<f32>
    %29 = stablehlo.reduce(%26 init: %27) applies stablehlo.maximum across dimensions = [2] : (tensor<1x2x2xf32>, tensor<f32>) -> tensor<1x2xf32>
    %30 = stablehlo.broadcast_in_dim %29, dims = [0, 1] : (tensor<1x2xf32>) -> tensor<1x2x2xf32>
    %31 = stablehlo.subtract %26, %30 : tensor<1x2x2xf32>
    %32 = stablehlo.exponential %31 : tensor<1x2x2xf32>
    %33 = stablehlo.reduce(%32 init: %28) applies stablehlo.add across dimensions = [2] : (tensor<1x2x2xf32>, tensor<f32>) -> tensor<1x2xf32>
    %34 = stablehlo.broadcast_in_dim %33, dims = [0, 1] : (tensor<1x2xf32>) -> tensor<1x2x2xf32>
    %35 = stablehlo.divide %32, %34 : tensor<1x2x2xf32>
    %36 = stablehlo.dot_general %35, %4, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<1x2x2xf32>, tensor<1x2x4xf32>) -> tensor<1x2x4xf32>
    %37 = stablehlo.reshape %36 : (tensor<1x2x4xf32>) -> tensor<2x4xf32>
    %38 = stablehlo.add %arg0, %37 : tensor<2x4xf32>
    %39 = stablehlo.multiply %38, %38 : tensor<2x4xf32>
    %40 = stablehlo.constant dense<0.0> : tensor<f32>
    %41 = stablehlo.reduce(%39 init: %40) applies stablehlo.add across dimensions = [1] : (tensor<2x4xf32>, tensor<f32>) -> tensor<2xf32>
    %42 = stablehlo.reshape %41 : (tensor<2xf32>) -> tensor<2x1xf32>
    %43 = stablehlo.constant dense<4.0> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %43, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
    %45 = stablehlo.divide %42, %44 : tensor<2x1xf32>
    %46 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
    %47 = stablehlo.add %45, %46 : tensor<2x1xf32>
    %48 = stablehlo.sqrt %47 : tensor<2x1xf32>
    %49 = stablehlo.broadcast_in_dim %48, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x4xf32>
    %50 = stablehlo.divide %38, %49 : tensor<2x4xf32>
    %51 = stablehlo.broadcast_in_dim %0, dims = [1] : (tensor<4xf32>) -> tensor<2x4xf32>
    %52 = stablehlo.multiply %50, %51 : tensor<2x4xf32>
    %53 = stablehlo.negate %52 : tensor<2x4xf32>
    %54 = stablehlo.exponential %53 : tensor<2x4xf32>
    %55 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<f32>) -> tensor<2x4xf32>
    %56 = stablehlo.add %55, %54 : tensor<2x4xf32>
    %57 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<f32>) -> tensor<2x4xf32>
    %58 = stablehlo.divide %57, %56 : tensor<2x4xf32>
    %59 = stablehlo.multiply %52, %58 : tensor<2x4xf32>
    %60 = stablehlo.multiply %59, %52 : tensor<2x4xf32>
    %61 = stablehlo.add %38, %60 : tensor<2x4xf32>
    return %61 : tensor<2x4xf32>
  }
}
