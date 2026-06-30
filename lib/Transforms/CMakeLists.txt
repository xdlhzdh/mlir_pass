set(LLVM_TARGET_DEFINITIONS ${PROJECT_SOURCE_DIR}/include/AICompiler/Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name AICompiler)
add_public_tablegen_target(AICompilerPassIncGen)

add_library(AICompilerTransforms STATIC
  ConvBNFusion.cpp
  ConvBNReluFusion.cpp
  SoftmaxLegalize.cpp
  RMSNormLegalize.cpp
  AttentionLegalize.cpp
  FlashAttentionTile.cpp
  RoPELegalize.cpp
  LayerNormLegalize.cpp
  GeluLegalize.cpp
  SwiGLULegalize.cpp
  QdqLegalize.cpp
  MatMulBiasFusion.cpp
  HorizontalGemmFusion.cpp
  ElementwiseChainLegalize.cpp
  ProducerConsumerLegalize.cpp
  LayoutBridgeLegalize.cpp
  LayoutFold.cpp
  KVCacheLegalize.cpp
  BroadcastSimplify.cpp
  StablehloConstantFold.cpp
  CustomLinalgOpt.cpp
  CustomLinalgTile.cpp
  CustomBufferOpt.cpp
  CustomLinalgToParallelLoops.cpp
  CustomLoopTiling.cpp
  CustomAffineOpt.cpp
  CustomVectorOpt.cpp
  CustomLLVMCleanup.cpp
  StableHLOToLinalg.cpp
  LinalgOptimization.cpp
  Bufferization.cpp
  LowerToLoops.cpp
  LowerToAffine.cpp
  LowerToVectorDialect.cpp
  Vectorization.cpp
  LLVMLowering.cpp
  RegisterPasses.cpp
)
add_dependencies(AICompilerTransforms AICompilerPassIncGen)

target_include_directories(AICompilerTransforms PUBLIC
  ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_BINARY_DIR}
)

target_link_libraries(AICompilerTransforms PUBLIC
  MLIRArithDialect
  MLIRBufferizationPipelines
  MLIRBufferizationTransforms
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRLinalgUtils
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRRewrite
  MLIRSCFDialect
  MLIRSCFToControlFlow
  MLIRSupport
  MLIRTransforms
  MLIRArithToLLVM
  MLIRAffineDialect
  MLIRAffineToStandard
  MLIRAffineTransforms
  MLIRAffineUtils
  MLIRControlFlowToLLVM
  MLIRFuncToLLVM
  MLIRIndexToLLVM
  MLIRMathToLLVM
  MLIRMemRefToLLVM
  MLIRBufferizationToMemRef
  MLIRLinalgToStandard
  MLIRVectorDialect
  MLIRVectorToLLVM
  MLIRReconcileUnrealizedCasts
)

target_include_directories(AICompilerTransforms PRIVATE
  ${STABLEHLO_INCLUDE_DIR}
)
