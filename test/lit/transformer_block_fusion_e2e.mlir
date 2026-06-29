// RUN: %pipe-demo --input=%s --pipeline-stop-after=fusion 2>&1 | FileCheck %s
// CHECK: aicom.rmsnorm_canonicalized
// CHECK: aicom.scaled_dot_product_attention
// CHECK: aicom.swiglu_canonicalized

module {
  func.func @inference() -> tensor<2x4xf32> {
    %x = stablehlo.constant dense<[[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]]> : tensor<2x4xf32>
    %weight = stablehlo.constant dense<[1.0, 1.0, 1.0, 1.0]> : tensor<4xf32>
    %eps = stablehlo.constant dense<1.0e-05> : tensor<f32>
    %four = stablehlo.constant dense<4.0> : tensor<f32>
    %zero = stablehlo.constant dense<0.0> : tensor<f32>
    %neg_inf = stablehlo.constant dense<-3.402823e+38> : tensor<f32>
    %scale = stablehlo.constant dense<0.5> : tensor<f32>
    %one = stablehlo.constant dense<1.0> : tensor<f32>

    // Pre-LN RMSNorm
    %x2_1 = stablehlo.multiply %x, %x : tensor<2x4xf32>
    %mean_1 = stablehlo.reduce(%x2_1 init: %zero) applies stablehlo.add across dimensions = [1]
        : (tensor<2x4xf32>, tensor<f32>) -> tensor<2xf32>
    %mean_1_r = stablehlo.reshape %mean_1 : (tensor<2xf32>) -> tensor<2x1xf32>
    %four_b = stablehlo.broadcast_in_dim %four, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
    %var_1 = stablehlo.divide %mean_1_r, %four_b : tensor<2x1xf32>
    %eps_b1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
    %var_eps_1 = stablehlo.add %var_1, %eps_b1 : tensor<2x1xf32>
    %denom_1 = stablehlo.sqrt %var_eps_1 : tensor<2x1xf32>
    %denom_1_b = stablehlo.broadcast_in_dim %denom_1, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x4xf32>
    %norm1 = stablehlo.divide %x, %denom_1_b : tensor<2x4xf32>
    %w_b1 = stablehlo.broadcast_in_dim %weight, dims = [1] : (tensor<4xf32>) -> tensor<2x4xf32>
    %norm1_w = stablehlo.multiply %norm1, %w_b1 : tensor<2x4xf32>

    // Attention on reshaped Q [1,2,4]
    %q = stablehlo.reshape %norm1_w : (tensor<2x4xf32>) -> tensor<1x2x4xf32>
    %kt = stablehlo.constant dense<[[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]]> : tensor<1x4x2xf32>
    %v = stablehlo.constant dense<[[[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]]]> : tensor<1x2x4xf32>
    %scores = stablehlo.dot_general %q, %kt,
        batching_dims = [0] x [0], contracting_dims = [2] x [1]
        : (tensor<1x2x4xf32>, tensor<1x4x2xf32>) -> tensor<1x2x2xf32>
    %scale_b = stablehlo.broadcast_in_dim %scale, dims = [] : (tensor<f32>) -> tensor<1x2x2xf32>
    %scaled = stablehlo.multiply %scores, %scale_b : tensor<1x2x2xf32>
    %max = "stablehlo.reduce"(%scaled, %neg_inf) ({
    ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
      %m = stablehlo.maximum %lhs, %rhs : tensor<f32>
      "stablehlo.return"(%m) : (tensor<f32>) -> ()
    }) { dimensions = array<i64: 2> } : (tensor<1x2x2xf32>, tensor<f32>) -> tensor<1x2xf32>
    %max_b = stablehlo.broadcast_in_dim %max, dims = [0, 1] : (tensor<1x2xf32>) -> tensor<1x2x2xf32>
    %shifted = stablehlo.subtract %scaled, %max_b : tensor<1x2x2xf32>
    %exp_a = stablehlo.exponential %shifted : tensor<1x2x2xf32>
    %sum_a = "stablehlo.reduce"(%exp_a, %zero) ({
    ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
      %s = stablehlo.add %lhs, %rhs : tensor<f32>
      "stablehlo.return"(%s) : (tensor<f32>) -> ()
    }) { dimensions = array<i64: 2> } : (tensor<1x2x2xf32>, tensor<f32>) -> tensor<1x2xf32>
    %sum_b_a = stablehlo.broadcast_in_dim %sum_a, dims = [0, 1] : (tensor<1x2xf32>) -> tensor<1x2x2xf32>
    %probs = stablehlo.divide %exp_a, %sum_b_a : tensor<1x2x2xf32>
    %attn = stablehlo.dot_general %probs, %v,
        batching_dims = [0] x [0], contracting_dims = [2] x [1]
        : (tensor<1x2x2xf32>, tensor<1x2x4xf32>) -> tensor<1x2x4xf32>
    %attn_2d = stablehlo.reshape %attn : (tensor<1x2x4xf32>) -> tensor<2x4xf32>
    %h = stablehlo.add %x, %attn_2d : tensor<2x4xf32>

    // Post-attention RMSNorm + SwiGLU (gate=up=norm)
    %x2_2 = stablehlo.multiply %h, %h : tensor<2x4xf32>
    %mean_2 = stablehlo.reduce(%x2_2 init: %zero) applies stablehlo.add across dimensions = [1]
        : (tensor<2x4xf32>, tensor<f32>) -> tensor<2xf32>
    %mean_2_r = stablehlo.reshape %mean_2 : (tensor<2xf32>) -> tensor<2x1xf32>
    %var_2 = stablehlo.divide %mean_2_r, %four_b : tensor<2x1xf32>
    %eps_b2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
    %var_eps_2 = stablehlo.add %var_2, %eps_b2 : tensor<2x1xf32>
    %denom_2 = stablehlo.sqrt %var_eps_2 : tensor<2x1xf32>
    %denom_2_b = stablehlo.broadcast_in_dim %denom_2, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x4xf32>
    %norm2 = stablehlo.divide %h, %denom_2_b : tensor<2x4xf32>
    %w_b2 = stablehlo.broadcast_in_dim %weight, dims = [1] : (tensor<4xf32>) -> tensor<2x4xf32>
    %norm2_w = stablehlo.multiply %norm2, %w_b2 : tensor<2x4xf32>

    %neg = stablehlo.negate %norm2_w : tensor<2x4xf32>
    %exp_s = stablehlo.exponential %neg : tensor<2x4xf32>
    %one_b = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<2x4xf32>
    %denom_s = stablehlo.add %one_b, %exp_s : tensor<2x4xf32>
    %sigmoid = stablehlo.divide %one_b, %denom_s : tensor<2x4xf32>
    %silu = stablehlo.multiply %norm2_w, %sigmoid : tensor<2x4xf32>
    %ffn = stablehlo.multiply %silu, %norm2_w : tensor<2x4xf32>
    %y = stablehlo.add %h, %ffn : tensor<2x4xf32>
    return %y : tensor<2x4xf32>
  }
}
