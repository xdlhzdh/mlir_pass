// RUN: %pipe-demo --input=%s --pipeline-stop-after=bufferize --loop-mode=scf-seq 2>&1 | FileCheck %s
//
// custom-buffer-opt may stack-promote the conv temporary, but the result
// buffer reaches func.return through memref.expand_shape and must stay on the
// heap (the JIT caller frees it).

// CHECK-LABEL: func.func @inference
// CHECK:       %[[TMP:.*]] = memref.alloca() {{.*}} : memref<1x2x2x2xf32>
// CHECK:       linalg.conv_2d_nhwc_hwcf {{.*}} outs(%[[TMP]]
// CHECK:       %[[OUT:.*]] = memref.alloc() {{.*}} : memref<2x2x2xf32>
// CHECK-NOT:   memref.alloca
// CHECK:       %[[RES:.*]] = memref.expand_shape %[[OUT]]
// CHECK:       return %[[RES]]

module {
  func.func @inference() -> tensor<1x2x2x2xf32> {
    %input = stablehlo.constant dense<[[[[1.0], [2.0]], [[3.0], [4.0]]]]> : tensor<1x2x2x1xf32>
    %weight = stablehlo.constant dense<[[[[0.5, 0.5]]]]> : tensor<1x1x1x2xf32>
    %zero = stablehlo.constant dense<0.0> : tensor<1x2x2x2xf32>
    %conv = stablehlo.convolution(%input, %weight)
        dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
        window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1],
                  rhs_dilate = [1, 1], reverse = [0, 0]}
        {batch_group_count = 1 : i64, feature_group_count = 1 : i64,
         precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]}
        : (tensor<1x2x2x1xf32>, tensor<1x1x1x2xf32>) -> tensor<1x2x2x2xf32>
    %relu = stablehlo.maximum %conv, %zero : tensor<1x2x2x2xf32>
    return %relu : tensor<1x2x2x2xf32>
  }
}
