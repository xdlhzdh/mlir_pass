module {
  func.func @inference() -> tensor<2x4xf32> {
    %x = stablehlo.constant dense<[[1.0, 2.0, 3.0, 4.0], [0.5, 1.5, 2.5, 3.5]]> : tensor<2x4xf32>
    %k = stablehlo.constant dense<1.702> : tensor<f32>
    %one = stablehlo.constant dense<1.0> : tensor<f32>
    %k_b = stablehlo.broadcast_in_dim %k, dims = [] : (tensor<f32>) -> tensor<2x4xf32>
    %one_b = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<2x4xf32>
    %kx = stablehlo.multiply %k_b, %x : tensor<2x4xf32>
    %neg = stablehlo.negate %kx : tensor<2x4xf32>
    %exp = stablehlo.exponential %neg : tensor<2x4xf32>
    %den = stablehlo.add %one_b, %exp : tensor<2x4xf32>
    %sig = stablehlo.divide %one_b, %den : tensor<2x4xf32>
    %out = stablehlo.multiply %x, %sig : tensor<2x4xf32>
    return %out : tensor<2x4xf32>
  }
}
