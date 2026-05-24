// MatMul + Add mini graph with embedded constants (JIT-friendly nullary function).
module {
  func.func @inference() -> tensor<2x2xf32> {
    %a = stablehlo.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
    %b = stablehlo.constant dense<[[1.0, 0.0], [0.0, 1.0]]> : tensor<2x2xf32>
    %bias = stablehlo.constant dense<[[0.5, 0.5], [0.5, 0.5]]> : tensor<2x2xf32>
    %mm = stablehlo.dot_general %a, %b,
      contracting_dims = [1] x [0],
      precision = [DEFAULT, DEFAULT]
      : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    %out = stablehlo.add %mm, %bias : tensor<2x2xf32>
    return %out : tensor<2x2xf32>
  }
}
