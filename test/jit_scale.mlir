// Nullary scale for JIT golden (2x constant multiply).
module {
  func.func @inference() -> tensor<2x2xf32> {
    %a = stablehlo.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
    %two = stablehlo.constant dense<2.0> : tensor<f32>
    %two_b = stablehlo.broadcast_in_dim %two, dims = [] : (tensor<f32>) -> tensor<2x2xf32>
    %out = stablehlo.multiply %a, %two_b : tensor<2x2xf32>
    return %out : tensor<2x2xf32>
  }
}
