// RUN: %pipe-demo --input=%s --pipeline-stop-after=fusion 2>&1 | FileCheck %s
// CHECK: scf.while

module {
  func.func @inference() -> tensor<4xf32> {
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    %init = arith.constant dense<0.0> : tensor<4xf32>
    %inc = arith.constant dense<1.0> : tensor<4xf32>
    %state:2 = scf.while (%arg0 = %c0, %arg1 = %init) : (index, tensor<4xf32>) -> (index, tensor<4xf32>) {
      %cond = arith.cmpi slt, %arg0, %c3 : index
      scf.condition(%cond) %arg0, %arg1 : index, tensor<4xf32>
    } do {
    ^bb0(%step: index, %acc: tensor<4xf32>):
      %step1 = arith.addi %step, %c1 : index
      %next = arith.addf %acc, %inc : tensor<4xf32>
      scf.yield %step1, %next : index, tensor<4xf32>
    }
    return %state#1 : tensor<4xf32>
  }
}
