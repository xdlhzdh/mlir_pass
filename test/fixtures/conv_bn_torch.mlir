module {
  func.func @inference(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<i64>, %arg3: tensor<1x3x32x32xf32>) -> tensor<1x8x30x30xf32> {
    %cst = stablehlo.constant dense_resource<torch_tensor_8_torch.float32_2> : tensor<8xf32>
    %cst_0 = stablehlo.constant dense_resource<torch_tensor_8_torch.float32_1> : tensor<8xf32>
    %cst_1 = stablehlo.constant dense_resource<torch_tensor_8_3_3_3_torch.float32> : tensor<8x3x3x3xf32>
    %cst_2 = stablehlo.constant dense_resource<torch_tensor_8_torch.float32> : tensor<8xf32>
    %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<8xf32>
    %cst_4 = arith.constant dense<1.000000e-05> : tensor<1xf64>
    %cst_5 = arith.constant dense<1> : tensor<1xi64>
    %0 = stablehlo.convolution(%arg3, %cst_1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x3x32x32xf32>, tensor<8x3x3x3xf32>) -> tensor<1x8x30x30xf32>
    %1 = stablehlo.reshape %cst_2 : (tensor<8xf32>) -> tensor<8x1x1xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [1, 2, 3] : (tensor<8x1x1xf32>) -> tensor<1x8x30x30xf32>
    %3 = stablehlo.add %0, %2 : tensor<1x8x30x30xf32>
    %4 = stablehlo.convert %cst_4 : (tensor<1xf64>) -> tensor<1xf32>
    %5 = stablehlo.reshape %4 : (tensor<1xf32>) -> tensor<f32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %7 = stablehlo.add %arg1, %6 : tensor<8xf32>
    %8 = stablehlo.sqrt %7 : tensor<8xf32>
    %9 = stablehlo.divide %cst_3, %8 : tensor<8xf32>
    %10 = stablehlo.convert %cst_5 : (tensor<1xi64>) -> tensor<1xf32>
    %11 = stablehlo.reshape %10 : (tensor<1xf32>) -> tensor<f32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %13 = stablehlo.multiply %9, %12 : tensor<8xf32>
    %14 = stablehlo.reshape %arg0 : (tensor<8xf32>) -> tensor<8x1xf32>
    %15 = stablehlo.reshape %14 : (tensor<8x1xf32>) -> tensor<8x1x1xf32>
    %16 = stablehlo.reshape %13 : (tensor<8xf32>) -> tensor<8x1xf32>
    %17 = stablehlo.reshape %16 : (tensor<8x1xf32>) -> tensor<8x1x1xf32>
    %18 = stablehlo.broadcast_in_dim %15, dims = [1, 2, 3] : (tensor<8x1x1xf32>) -> tensor<1x8x30x30xf32>
    %19 = stablehlo.subtract %3, %18 : tensor<1x8x30x30xf32>
    %20 = stablehlo.broadcast_in_dim %17, dims = [1, 2, 3] : (tensor<8x1x1xf32>) -> tensor<1x8x30x30xf32>
    %21 = stablehlo.multiply %19, %20 : tensor<1x8x30x30xf32>
    %22 = stablehlo.reshape %cst_0 : (tensor<8xf32>) -> tensor<8x1xf32>
    %23 = stablehlo.reshape %22 : (tensor<8x1xf32>) -> tensor<8x1x1xf32>
    %24 = stablehlo.broadcast_in_dim %23, dims = [1, 2, 3] : (tensor<8x1x1xf32>) -> tensor<1x8x30x30xf32>
    %25 = stablehlo.multiply %21, %24 : tensor<1x8x30x30xf32>
    %26 = stablehlo.reshape %cst : (tensor<8xf32>) -> tensor<8x1xf32>
    %27 = stablehlo.reshape %26 : (tensor<8x1xf32>) -> tensor<8x1x1xf32>
    %28 = stablehlo.broadcast_in_dim %27, dims = [1, 2, 3] : (tensor<8x1x1xf32>) -> tensor<1x8x30x30xf32>
    %29 = stablehlo.add %25, %28 : tensor<1x8x30x30xf32>
    return %29 : tensor<1x8x30x30xf32>
  }
}

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_8_torch.float32_2: "0x040000000000000000000000000000000000000000000000000000000000000000000000",
      torch_tensor_8_torch.float32_1: "0x040000000000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F",
      torch_tensor_8_3_3_3_torch.float32: "0x04000000C989A1BDCF46F13DBCE008BC13CAC53D31A9B73DA747403E6BC43DBEA7BB98BC1A60BD3D37B12FBE15E6F43C5260413AE4C21B3E7E7C96BD0CE104BEA5638A3D98533F3E221723BEE329053E2CEE3C3D6EA0D8BDF5D3FD3CE1D6893BC88D1DBE96074DBDFCAE9D3CA37E9E3ACD48D9BDD2A618BEC9AD1ABE694EE13D6D5F853DAADEA8BD092E353D0DD6E63DAA9BC93D0F676CBD21C3A93DFFAC163ECB78C9BD25BDBABD5634EC3D05DA3EBE96DE31BE34E43EBD8A4D693C1996CD3DE12727BE7DD818BEEC5FF6BDFD61EB3D5257D63C6C52263E59AD123E7A4F41BE1EB8B5BD00FEF53D6DAC193E9D01173E9BA50FBD9960E33C5C11FE3D40C33F3D25B4CABD048D04BD57D6BD3D9CBAC53D3372B3BDADE4253EDBB2B4BD78EE263EF9E839BE7D4F40BEBB43CABDDD7A663D1B31393E03C9E4BC8F022BBEF3AD1CBE8CE5E5BA065A043E0F3D4CBDC545BBBD690773BDFF2E2A3D5BFBE73C287F2A3E7C8A0B3E87092A3E5809C8BD868195BD0DCCB53D3ECE443EC265D33DE74F8F3D06B8403D15D9C8BBC96A84BDD08B1B3EE85D433E4896453D93E10CBEB14D093E42CAE7BD1E84CFBCD61A86BC9C4341BEC21AA73D7486FE3D1EC61ABE865E9E3D9459253E6CB53CBE762B8A3DC7672B3E0A10B5BC498C4A3D8D892EBE75F58DBD3241E73D6B3B0BBDCE8D8ABCD3A6273DE33141BE043A3C3DA09003BE8B0F1CBE5945603D145E4BBD460FD8BDD8F740BE399B36BED6BF383D0CB73B3E773A2BBE211426BE9283A2BC1A3A1EBD8C1D31BE772E37BE5DA83F3E0A36783D7EA9AEBD27CE91BDC715CBBD0029F63D3B8D133E9C8D4E3D12A60DBD755502BEAF8EE4BDBAA593BD3C6EADBBAC6F4C3DFBF32A3EA3E13F3D71161B3EB5BE2BBE466932BE9E5311BE17F868BD3AF84ABDE07F053E3E919CBDB8D601BE27942A3D6383BABD6839213ED117C33D0DE1A7BDE6CF7E3D36CFD33D1F142F3E487188BD774B53BC729242BE972FD0BA711BE73C5C3D55BDB3D3B33C4D559B3D4C35003ED7AD463DF4723EBEC4470E3ECC9116BE4BF7FDBD360FA7BD47FE253EA4EB3F3CE3A2AC3DB442A7BD48C7DFBD70342EBD9A3AA8BD3C3C08BE30F034BE3C392F3EC4D0143EEDB2E4BC1FFCDA3D505BCB3DA01B1C3E0EB85ABDC70CA53DAD61323EF7D8613DC64DF83DBF8F033E4892993CA9CEBCBC25E010BE6D10343EE5F95DBB44651A3E0D435BBD",
      torch_tensor_8_torch.float32: "0x04000000F08A1F3E17D7253EDA3BA4BD90263EBEABB70E3E56C635BC6FE69F3D5D7429BE"
    }
  }
#-}
