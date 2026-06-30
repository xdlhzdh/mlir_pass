module {
  func.func @inference() -> tensor<2x4xf32> {
    %x = stablehlo.constant dense<[[1.0, 2.0, 3.0, 4.0], [0.5, 1.5, 2.5, 3.5]]> : tensor<2x4xf32>
    %half = stablehlo.constant dense<0.5> : tensor<f32>
    %one = stablehlo.constant dense<1.0> : tensor<f32>
    %two = stablehlo.constant dense<2.0> : tensor<f32>
    %sqrt2 = stablehlo.sqrt %two : tensor<f32>
    %sqrt2_b = stablehlo.broadcast_in_dim %sqrt2, dims = [] : (tensor<f32>) -> tensor<2x4xf32>
    %div = stablehlo.divide %x, %sqrt2_b : tensor<2x4xf32>
    %erf = chlo.erf %div : tensor<2x4xf32> -> tensor<2x4xf32>
    %one_b = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<2x4xf32>
    %inner = stablehlo.add %one_b, %erf : tensor<2x4xf32>
    %half_b = stablehlo.broadcast_in_dim %half, dims = [] : (tensor<f32>) -> tensor<2x4xf32>
    %half_x = stablehlo.multiply %half_b, %x : tensor<2x4xf32>
    %out = stablehlo.multiply %half_x, %inner : tensor<2x4xf32>
    return %out : tensor<2x4xf32>
  }
}
