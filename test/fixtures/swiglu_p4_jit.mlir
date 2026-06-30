module {
  func.func @inference() -> tensor<2x4xf32> {
    %gate = stablehlo.constant dense<[[1.0, 0.0, -1.0, 2.0], [0.5, 1.0, 1.5, 2.0]]> : tensor<2x4xf32>
    %up = stablehlo.constant dense<[[2.0, 1.0, 0.5, 0.25], [1.0, 1.0, 1.0, 1.0]]> : tensor<2x4xf32>
    %one = stablehlo.constant dense<1.0> : tensor<f32>
    %one_b = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<2x4xf32>
    %neg = stablehlo.negate %gate : tensor<2x4xf32>
    %exp = stablehlo.exponential %neg : tensor<2x4xf32>
    %den = stablehlo.add %one_b, %exp : tensor<2x4xf32>
    %silu = stablehlo.divide %gate, %den : tensor<2x4xf32>
    %out = stablehlo.multiply %silu, %up : tensor<2x4xf32>
    return %out : tensor<2x4xf32>
  }
}
